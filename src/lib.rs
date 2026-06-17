pub mod vector_store;

use anyhow::Result;
use notify::{Event as NotifyEvent, RecursiveMode, Watcher};
use rusqlite::OptionalExtension;
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_rusqlite::Connection;

use crate::vector_store::VectorStore;

const DEBOUNCE_MS: u64 = 200;

#[derive(Debug, Clone)]
pub enum EventType {
    Create,
    Modify,
    Delete,
}

#[derive(Debug, Clone)]
pub struct FileEvent {
    pub file_path: String,
    pub etype: EventType,
}

pub struct DocumentStore<const D: usize> {
    pub conn: Connection,
    watcher: notify::RecommendedWatcher,
    rx: mpsc::Receiver<FileEvent>,
    pub vs: VectorStore<D>,
}

impl<const D: usize> DocumentStore<D> {
    pub async fn new(db_path: &str, vs: VectorStore<D>) -> Result<Self, anyhow::Error> {
        let conn = Connection::open(db_path).await?;

        conn.call(|conn| conn.execute("PRAGMA foreign_keys = ON;", []))
            .await?;

        conn.call(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    file_hash BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    eid INTEGER PRIMARY KEY,
                    file TEXT NOT NULL,
                    FOREIGN KEY (file) REFERENCES file_hashes(file_path) ON DELETE CASCADE
                );",
            )
        })
        .await?;

        let (tx, rx) = mpsc::channel(256);

        let debounce_logs: Arc<Mutex<HashSet<PathBuf>>> = Arc::new(Mutex::new(HashSet::new()));
        let debounce_logs2 = debounce_logs.clone();
        let tokio_r = tokio::runtime::Handle::current();

        let watcher = notify::recommended_watcher(move |res: notify::Result<NotifyEvent>| {
            if let Ok(mut event) = res {
                let mut debl = debounce_logs2.lock().unwrap();
                event.paths.retain(|path| {
                    if debl.contains(path) {
                        return false;
                    } else {
                        debl.insert(path.clone());
                        return true;
                    }
                });

                let kind = event.kind;

                for path in event.paths {
                    let debounce_logs3 = debounce_logs2.clone();
                    let tx_clone = tx.clone();
                    tokio_r.spawn(async move {
                        tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)).await;
                        debounce_logs3.lock().unwrap().remove(&path);

                        let etype = match kind {
                            notify::EventKind::Create(_) => Some(EventType::Create),
                            notify::EventKind::Modify(_) => Some(EventType::Modify),
                            notify::EventKind::Remove(_) => Some(EventType::Delete),
                            _ => None,
                        };
                        if let Some(et) = etype {
                            let _ = tx_clone
                                .send(FileEvent {
                                    file_path: std::path::absolute(&path)?
                                        .to_str()
                                        .expect(&format!(
                                            "path is not valid utf-8: {}",
                                            path.display(),
                                        ))
                                        .to_string(),
                                    etype: et.clone(),
                                })
                                .await;
                        }
                        Ok::<(), anyhow::Error>(())
                    });
                }
            }
        })?;

        let mut store = Self {
            conn,
            watcher: watcher,
            rx,
            vs,
        };

        // Purge files that no longer exist
        let all_files = store
            .conn
            .call(|conn| {
                let mut stmt = conn.prepare("SELECT file_path FROM file_hashes")?;
                let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
                let mut files = Vec::new();
                for row in rows {
                    files.push(row?);
                }
                Ok::<_, rusqlite::Error>(files)
            })
            .await?;

        for file in all_files {
            if !Path::new(&file).exists() {
                store.delete_file(&file).await?;
            }
        }

        Ok(store)
    }

    pub async fn add_directory(&mut self, dir: &str) -> Result<Vec<PathBuf>, anyhow::Error> {
        self.watcher
            .watch(Path::new(dir), RecursiveMode::Recursive)?;

        self.index_directory_files(dir).await
    }

    pub async fn next_event(&mut self) -> Option<FileEvent> {
        loop {
            if let Some(ev) = self.rx.recv().await {
                let should_return = match ev.etype {
                    EventType::Create | EventType::Modify => self
                        .gen_content_hash(Path::new(&ev.file_path))
                        .await
                        .unwrap_or(false),
                    EventType::Delete => {
                        let path = ev.file_path.clone();
                        self.conn
                            .call(move |conn| {
                                conn.execute(
                                    "DELETE FROM file_hashes WHERE file_path = ?1",
                                    rusqlite::params![path],
                                )
                            })
                            .await
                            .is_ok()
                    }
                };

                if should_return {
                    return Some(ev);
                }
            }
        }
    }

    async fn index_directory_files(&self, dir: &str) -> Result<Vec<PathBuf>, anyhow::Error> {
        let mut files = Vec::new();
        let walker = ignore::WalkBuilder::new(dir).hidden(false).build();

        for result in walker {
            if let Ok(entry) = result {
                if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    continue;
                }

                let path = entry.path().to_string_lossy().to_string();
                let mut path_rslv = std::path::absolute(path)?;
                path_rslv = std::path::absolute(path_rslv)?;

                if self.gen_content_hash(&path_rslv).await.unwrap_or(false) {
                    files.push(path_rslv);
                }
            }
        }

        Ok(files)
    }

    pub async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        let vector_result = self.vs.search(query);
        if vector_result.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<u64> = vector_result.iter().map(|(id, _)| *id).collect();
        let ids_str = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT eid, file FROM embeddings WHERE eid IN ({})",
            ids_str
        );

        let files_map = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&sql)?;
                let mut id_to_file = HashMap::new();

                let rows = stmt.query_map([], |row| {
                    let eid: u64 = row.get(0)?;
                    let file: String = row.get(1)?;
                    Ok((eid, file))
                })?;

                for row in rows {
                    if let Ok((eid, file)) = row {
                        id_to_file.insert(eid, file);
                    }
                }

                Ok::<_, rusqlite::Error>(id_to_file)
            })
            .await?;

        let mut unique_files = Vec::new();
        let mut seen = HashSet::new();

        for (id, score) in vector_result {
            if let Some(file) = files_map.get(&id) {
                if seen.insert(file.clone()) {
                    unique_files.push((file.clone(), score));
                    if unique_files.len() == top_k {
                        break;
                    }
                }
            }
        }

        Ok(unique_files)
    }

    pub async fn embed_file<F>(&mut self, path: String, fetch_embeddings: F) -> Result<()>
    where
        F: AsyncFn(&[&str]) -> Result<Vec<Vec<f32>>>,
    {
        let content = tokio::fs::read_to_string(&path).await?;
        let chunks = self.vs.chunk_text(path, &content);
        let prev_node_count = self.vs.node_count();
        if !chunks.is_empty() {
            let texts: Vec<&str> = chunks.iter().map(|s| s.text.as_str()).collect();
            let embeddings = fetch_embeddings(&texts).await?;
            self.vs.add_embeddings(&chunks, &embeddings).await?;
            self.conn
                .call(move |con| {
                    let tx = con.transaction()?;
                    let mut stm =
                        tx.prepare("INSERT INTO embeddings (eid, file) VALUES (?1, ?2);")?;
                    for (i, chunk) in chunks.iter().enumerate() {
                        let eid = prev_node_count + i as u64 + 1;
                        stm.execute(rusqlite::params![eid, chunk.file_path])?;
                    }
                    drop(stm);
                    tx.commit()
                })
                .await?;
        }
        Ok(())
    }

    pub async fn delete_file(&mut self, path: &str) -> Result<()> {
        let sql = "SELECT eid FROM embeddings WHERE file = ?1";

        let path_clone = path.to_string();
        let eids: Vec<u64> = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(sql)?;
                let rows = stmt.query_map([path_clone], |row| row.get(0))?;

                let mut ids = Vec::new();
                for row in rows {
                    ids.push(row?);
                }
                Ok::<_, rusqlite::Error>(ids)
            })
            .await?;

        self.vs.delete_nodes(&eids);

        let path_clone = path.to_string();
        self.conn
            .call(move |conn| {
                conn.execute("DELETE FROM file_hashes WHERE file_path = ?1", [path_clone])?;
                Ok::<_, rusqlite::Error>(())
            })
            .await?;

        Ok(())
    }

    async fn gen_content_hash(&self, path: &Path) -> Result<bool, anyhow::Error> {
        let content = tokio::fs::read(path).await?;
        let mut hasher = Sha1::new();
        hasher.update(&content);
        let hash = hasher.finalize().to_vec();

        let path2 = path
            .to_str()
            .expect(&format!("path is not valid utf-8: {}", path.display()))
            .to_string();
        let path3 = path2.clone();
        let hash_clone1 = hash.clone();

        let row_res = self
            .conn
            .call(move |conn| {
                let mut stmt =
                    conn.prepare("SELECT file_hash FROM file_hashes WHERE file_path = ?1")?;
                let prev_hash_opt: Option<Vec<u8>> = stmt
                    .query_row(rusqlite::params![path2], |row| row.get(0))
                    .optional()?;
                Ok::<Option<Vec<u8>>, rusqlite::Error>(prev_hash_opt)
            })
            .await?;

        if let Some(prev_hash) = row_res {
            if prev_hash != hash {
                self.conn
                    .call(move |conn| {
                        conn.execute(
                            "UPDATE file_hashes SET file_hash = ?1 WHERE file_path = ?2",
                            rusqlite::params![hash_clone1, path3],
                        )
                    })
                    .await?;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            self.conn
                .call(move |conn| {
                    conn.execute(
                        "INSERT INTO file_hashes (file_path, file_hash) VALUES (?1, ?2)",
                        rusqlite::params![path3, hash_clone1],
                    )
                })
                .await?;
            Ok(true)
        }
    }
}
