use bytemuck::{Pod, Zeroable};
use memmap2::{MmapMut, RemapOptions};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncReadExt;

pub struct Chunk {
    pub chunk_id: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub text: String,
}

#[derive(Clone, Copy)]
pub struct VectorStoreConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub m: usize,
    pub m0: usize,
    pub ef_cons: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 200,
            overlap: 50,
            m: 16,
            m0: 32,
            ef_cons: 250,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut v_dst = _mm256_setzero_ps();
    let mut v_norm_a = _mm256_setzero_ps();
    let mut v_norm_b = _mm256_setzero_ps();

    let chunks = a.len() / 8;
    for i in 0..chunks {
        let ia = unsafe { a.as_ptr().add(i * 8) };
        let ib = unsafe { b.as_ptr().add(i * 8) };

        let va = unsafe { _mm256_loadu_ps(ia) };
        let vb = unsafe { _mm256_loadu_ps(ib) };

        v_dst = _mm256_fmadd_ps(va, vb, v_dst);
        v_norm_a = _mm256_fmadd_ps(va, va, v_norm_a);
        v_norm_b = _mm256_fmadd_ps(vb, vb, v_norm_b);
    }

    let mut dot_arr = [0.0; 8];
    let mut norm_a_arr = [0.0; 8];
    let mut norm_b_arr = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(dot_arr.as_mut_ptr(), v_dst);
        _mm256_storeu_ps(norm_a_arr.as_mut_ptr(), v_norm_a);
        _mm256_storeu_ps(norm_b_arr.as_mut_ptr(), v_norm_b);
    }

    let mut dot = dot_arr.iter().sum::<f32>();
    let mut norm_a = norm_a_arr.iter().sum::<f32>();
    let mut norm_b = norm_b_arr.iter().sum::<f32>();

    let remainder = a.len() % 8;
    if remainder != 0 {
        for i in (a.len() - remainder)..a.len() {
            let x = a[i];
            let y = b[i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn cosine_similarity_default(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_similarity_avx2(a, b) };
        }
    }
    cosine_similarity_default(a, b)
}

pub struct VectorStore<const D: usize> {
    pub vec_path: String,
    pub config: VectorStoreConfig,
    pub mmap: MmapMut,
    pub file: File,
    pub est_max_layer: u8,
}

const MAGIC: [u8; 5] = [0x54, 0x56, 0x45, 0x43, 0x44];
const VERSION: [u8; 2] = [0x00, 0x02];

#[repr(C)]
#[derive(Pod, Clone, Copy, Zeroable)]
pub struct Header {
    pub magic: [u8; 5],
    pub _pad1: u8,
    pub version: [u8; 2],
    pub nodes: u64,
    pub entry: u64,
    pub vector_dim: u32,
    pub m: u32,
    pub m0: u32,
    pub max_layer: u32,
    pub ef_cons: u16,
    pub _pad2: [u8; 6],
}

const HEADER_SIZE: usize = size_of::<Header>();

impl<const D: usize> VectorStore<D> {
    pub async fn init(vec_path: &str, config: VectorStoreConfig) -> Result<Self, anyhow::Error> {
        let est_max_layer = (u32::ilog10(1_000_000) / u8::ilog10(config.m as u8)) as u8;
        let path = Path::new(vec_path);
        if !path.exists() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .await?;
            let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

            let mut store = Self {
                vec_path: vec_path.to_string(),
                config,
                mmap,
                file,
                est_max_layer,
            };
            store.write_header().await?;
            return Ok(store);
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await?;

        let mut magic_bytes = [0u8; 5];
        file.read_exact(&mut magic_bytes).await?;
        if magic_bytes != MAGIC {
            return Err(anyhow::anyhow!("Invalid magic number"));
        }
        _ = file.read_u8().await?;

        let mut version = [0u8; 2];
        file.read_exact(&mut version).await?;
        if version != VERSION {
            return Err(anyhow::anyhow!("Invalid version"));
        }

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        let mut store = Self {
            vec_path: vec_path.to_string(),
            config,
            mmap,
            file,
            est_max_layer,
        };
        store.verify_db().await?;
        Ok(store)
    }

    pub fn header_mut(&mut self) -> &mut Header {
        bytemuck::from_bytes_mut(&mut self.mmap[0..HEADER_SIZE])
    }

    pub fn header(&self) -> &Header {
        bytemuck::from_bytes(&self.mmap[0..HEADER_SIZE])
    }

    pub fn node_count(&self) -> u64 {
        self.header().nodes
    }

    pub fn entry(&self) -> Option<u64> {
        if self.header().entry == 0 {
            None
        } else {
            Some(self.header().entry)
        }
    }

    pub async fn write_header(&mut self) -> Result<(), anyhow::Error> {
        if self.mmap.len() < HEADER_SIZE {
            self.file.set_len(HEADER_SIZE as u64).await?;
            unsafe {
                self.mmap
                    .remap(HEADER_SIZE, RemapOptions::new().may_move(true))?;
            }
        }
        let header = Header {
            magic: MAGIC,
            _pad1: 0,
            version: VERSION,
            nodes: 0,
            entry: 0,
            vector_dim: D as u32,
            m: self.config.m as u32,
            m0: self.config.m0 as u32,
            max_layer: 0,
            ef_cons: self.config.ef_cons as u16,
            _pad2: [0u8; 6],
        };

        self.mmap[0..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));

        Ok(())
    }

    pub async fn verify_db(&mut self) -> Result<(), anyhow::Error> {
        let header: &Header = bytemuck::try_from_bytes(&self.mmap[0..HEADER_SIZE])
            .map_err(|_| anyhow::anyhow!("Invalid vector store header"))?;

        if header.vector_dim as usize != D {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        if header.m as usize != self.config.m {
            return Err(anyhow::anyhow!(
                "HNSW m mismatch: store has {}, config has {}",
                header.m,
                self.config.m
            ));
        }
        if header.m0 as usize != self.config.m0 {
            return Err(anyhow::anyhow!(
                "HNSW m0 mismatch: store has {}, config has {}",
                header.m0,
                self.config.m0
            ));
        }
        if header.ef_cons as usize != self.config.ef_cons {
            return Err(anyhow::anyhow!(
                "HNSW ef_construction mismatch: store has {}, config has {}",
                header.ef_cons,
                self.config.ef_cons
            ));
        }

        Ok(())
    }

    pub fn chunk_text(&self, file_path: String, text: &str) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut texts = Vec::new();

        let step = self.config.chunk_size - self.config.overlap;
        if words.is_empty() {
            return Vec::new();
        }

        let mut i = 0;
        let mut chunk_index = 0;
        while i < words.len() {
            let end = std::cmp::min(i + self.config.chunk_size, words.len());
            let chunk_words = &words[i..end];
            let join_text = chunk_words.join(" ");

            chunks.push(Chunk {
                chunk_id: format!("{}::{chunk_index}", &file_path),
                chunk_index,
                file_path: file_path.clone(),
                text: join_text.clone(),
            });
            texts.push(join_text);

            chunk_index += 1;
            if end == words.len() {
                break;
            }
            i += step;
        }

        chunks
    }

    fn random_layer(&self) -> u32 {
        let mut r: f64 = rand::random();

        let ml = 1.0 / (self.header().m as f64).ln();
        if r <= 0.0 {
            r = 0.5;
        }
        (-r.ln() * ml).floor() as u32
    }

    pub fn record_size(&self) -> usize {
        let m0 = self.header().m0 as usize;
        let m = self.header().m as usize;

        let base = (8 + D * size_of::<f32>() + 7) & !7;

        base + m0 * size_of::<u64>() + ((self.est_max_layer - 1) as usize) * m * size_of::<u64>()
    }

    pub fn node_offset(&self, eid: u64) -> usize {
        assert!(eid <= self.header().nodes);
        HEADER_SIZE + ((eid - 1) as usize) * self.record_size()
    }

    pub fn is_deleted(&self, eid: u64) -> bool {
        let offset = self.node_offset(eid);
        let raw_eid = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap());
        (raw_eid & (1 << 63)) != 0
    }

    pub fn set_deleted(&mut self, eid: u64, deleted: bool) {
        let offset = self.node_offset(eid);
        let mut raw_eid = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap());
        if deleted {
            raw_eid |= 1 << 63;
        } else {
            raw_eid &= !(1 << 63);
        }
        self.mmap[offset..offset + 8].copy_from_slice(&raw_eid.to_le_bytes());
    }

    pub fn vector(&self, eid: u64) -> &[f32] {
        let offset = self.node_offset(eid) + 8;
        bytemuck::cast_slice(&self.mmap[offset..offset + D * size_of::<f32>()])
    }

    fn connections_offset(&self, eid: u64, layer: usize) -> usize {
        let base = (8 + D * size_of::<f32>() + 7) & !7;
        let mut offset = self.node_offset(eid) + base;
        if layer > 0 {
            let m0 = self.header().m0 as usize;
            offset += m0 * size_of::<u64>();
            let m = self.header().m as usize;
            offset += (layer - 1) * m * size_of::<u64>();
        }
        offset
    }

    pub fn connections(&self, eid: u64, layer: usize) -> &[u64] {
        let offset = self.connections_offset(eid, layer);
        let count = if layer == 0 {
            self.header().m0 as usize
        } else {
            self.header().m as usize
        };
        bytemuck::cast_slice(&self.mmap[offset..offset + count * size_of::<u64>()])
    }

    pub fn connections_mut(&mut self, eid: u64, layer: usize) -> &mut [u64] {
        let offset = self.connections_offset(eid, layer);
        let count = if layer == 0 {
            self.header().m0 as usize
        } else {
            self.header().m as usize
        };
        bytemuck::cast_slice_mut(&mut self.mmap[offset..offset + count * size_of::<u64>()])
    }

    pub fn get_connections_vec(&self, eid: u64, layer: usize) -> Vec<u64> {
        self.connections(eid, layer)
            .iter()
            .copied()
            .filter(|&x| x != 0)
            .collect()
    }

    pub fn set_connections_vec(&mut self, eid: u64, layer: usize, conns: &[u64]) {
        let slice = self.connections_mut(eid, layer);
        for i in 0..slice.len() {
            if i < conns.len() {
                slice[i] = conns[i];
            } else {
                slice[i] = 0;
            }
        }
    }

    pub async fn add_embeddings(
        &mut self,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<(), anyhow::Error> {
        let prev_length = self.mmap.len();
        self.file
            .set_len((prev_length + chunks.len() * self.record_size()) as u64)
            .await?;

        unsafe {
            self.mmap.remap(
                prev_length + chunks.len() * self.record_size(),
                RemapOptions::new().may_move(true),
            )?;
        }

        for (_chunk, e) in chunks.iter().zip(embeddings.iter()) {
            let eid = self.header().nodes + 1;
            self.header_mut().nodes += 1;
            let node_layer = self.random_layer();
            assert!(node_layer < self.est_max_layer as u32);

            let offset = self.node_offset(eid);
            // eid with deleted flag (MSB is 0 initially)
            self.mmap[offset..offset + 8].copy_from_slice(&eid.to_le_bytes());
            let vec_bytes = bytemuck::cast_slice(e);
            self.mmap[offset + 8..offset + 8 + D * size_of::<f32>()].copy_from_slice(vec_bytes);

            // Cool trick that I learned: to round a number to nearest multiple of 8
            // 1. To round up the number add 7 to it otherwise keep it as it is
            // 2. Do a bitwise and with bitwise not of 7
            // What this does is zero out the last 3 bits of the number and the remaining number is
            // now guaranteed to be a multiple of 8
            // This let's us align the records at intervals of 8 preventing any unaligned casting
            // errors from bytemuck
            let base = (8 + D * size_of::<f32>() + 7) & !7;
            let conn_start = offset + base;
            let conn_end = offset + self.record_size();
            for b in self.mmap[conn_start..conn_end].iter_mut() {
                *b = 0;
            }

            let mut en = match self.entry() {
                Some(e) => e,
                None => {
                    let h = self.header_mut();
                    h.entry = eid;
                    h.max_layer = node_layer;
                    continue;
                }
            };

            let mut layer = self.header().max_layer;
            while layer > node_layer {
                let candidates = self.search_layer(e, en, 1, layer as usize);
                if let Some(best) = best_match(&candidates) {
                    en = best.id;
                }
                if layer == 0 {
                    break;
                }
                layer -= 1;
            }

            let mut layer_idx = std::cmp::min(node_layer, self.header().max_layer) as isize;
            while layer_idx >= 0 {
                let l = layer_idx as usize;
                let m = if l == 0 {
                    self.header().m0 as usize
                } else {
                    self.header().m as usize
                };
                let candidates = self.search_layer(e, en, self.header().ef_cons as usize, l);

                if let Some(best) = best_match(&candidates) {
                    en = best.id;
                }

                let neighbours = self.select_neighbours(&candidates, m);
                self.set_connections_vec(eid, l, &neighbours);

                for nb in neighbours {
                    let mut nb_conns = self.get_connections_vec(nb, l);
                    nb_conns.push(eid);
                    self.set_connections_vec(nb, l, &nb_conns);
                    if nb_conns.len() > m {
                        self.prune_connections(nb, l, m);
                    }
                }

                layer_idx -= 1;
            }

            if node_layer > self.header().max_layer {
                let h = self.header_mut();
                h.max_layer = node_layer;
                h.entry = eid;
            }
        }
        Ok(())
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> BinaryHeap<ResultItem> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut result = BinaryHeap::new();

        if !self.is_deleted(entry) {
            let d = cosine_similarity(query, self.vector(entry));
            visited.insert(entry);
            candidates.push(Candidate { sim: d, id: entry });
            result.push(ResultItem { sim: d, id: entry });
        } else {
            for nb in self.get_connections_vec(entry, layer) {
                if !self.is_deleted(nb) {
                    let d = cosine_similarity(query, self.vector(nb));
                    visited.insert(nb);
                    candidates.push(Candidate { sim: d, id: nb });
                    result.push(ResultItem { sim: d, id: nb });
                    break;
                }
            }
        }

        while let Some(candidate) = candidates.pop() {
            if result.len() >= ef
                && let Some(furthest) = result.peek()
                && candidate.sim < furthest.sim
            {
                break;
            }

            for neighbor in self.get_connections_vec(candidate.id, layer) {
                if self.is_deleted(neighbor) {
                    continue;
                }
                if visited.insert(neighbor) {
                    let similarity = cosine_similarity(query, self.vector(neighbor));

                    let mut should_push = true;
                    if result.len() >= ef
                        && let Some(furthest) = result.peek()
                        && similarity <= furthest.sim
                    {
                        should_push = false;
                    }

                    if should_push {
                        candidates.push(Candidate {
                            sim: similarity,
                            id: neighbor,
                        });
                        result.push(ResultItem {
                            sim: similarity,
                            id: neighbor,
                        });
                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        result
    }

    fn select_neighbours(&self, candidates: &BinaryHeap<ResultItem>, m: usize) -> Vec<u64> {
        let count = std::cmp::min(candidates.len(), m);
        let mut items: Vec<_> = candidates.clone().into_vec();
        items.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap_or(Ordering::Equal));

        items.into_iter().take(count).map(|c| c.id).collect()
    }

    fn prune_connections(&mut self, node_id: u64, layer: usize, m: usize) {
        let connections = self.get_connections_vec(node_id, layer);
        if connections.len() <= m {
            return;
        }

        let mut candidates = Vec::new();
        for neighbour_id in connections {
            let dist = cosine_similarity(self.vector(node_id), self.vector(neighbour_id));
            candidates.push((dist, neighbour_id));
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        candidates.truncate(m);

        let pruned: Vec<u64> = candidates.into_iter().map(|(_, id)| id).collect();
        self.set_connections_vec(node_id, layer, &pruned);
    }

    pub fn search(&self, query_vec: &[f32]) -> Vec<(u64, f32)> {
        let mut en = match self.entry() {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut layer = self.header().max_layer as isize;
        while layer > 0 {
            let l = layer as usize;
            let candidates = self.search_layer(query_vec, en, 1, l);
            if let Some(best) = best_match(&candidates) {
                en = best.id;
            }
            layer -= 1;
        }

        let result = self.search_layer(query_vec, en, 50usize, 0);

        let mut sorted_result: Vec<_> = result.into_vec();
        sorted_result.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap_or(Ordering::Equal));

        sorted_result.into_iter().map(|c| (c.id, c.sim)).collect()
    }

    pub fn delete_nodes(&mut self, nodes: &[u64]) {
        for n in nodes {
            self.set_deleted(*n, true);
        }
    }
}

fn best_match(heap: &BinaryHeap<ResultItem>) -> Option<ResultItem> {
    let items = heap.clone().into_vec();
    items
        .into_iter()
        .max_by(|a, b| a.sim.partial_cmp(&b.sim).unwrap_or(Ordering::Equal))
}

#[derive(Clone, Copy)]
struct Candidate {
    sim: f32,
    id: u64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.partial_cmp(&other.sim).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Copy)]
struct ResultItem {
    sim: f32,
    id: u64,
}

impl PartialEq for ResultItem {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim && self.id == other.id
    }
}

impl Eq for ResultItem {}

impl PartialOrd for ResultItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResultItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.sim.partial_cmp(&self.sim).unwrap_or(Ordering::Equal)
    }
}
