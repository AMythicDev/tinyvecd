use anyhow::Result;
use std::env;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

mod gemini;

use tinyvecd::{
    DocumentStore, EventType,
    vector_store::{VectorStore, VectorStoreConfig},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let gemini_api_key = env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow::anyhow!("Error: GEMINI_API_KEY environment variable is not set."))?;

    let vs_config = VectorStoreConfig::default();
    let vs = VectorStore::<3072>::init(".vec.tvecd", vs_config).await?;

    let mut ds = DocumentStore::new(".test-db", vs).await?;

    let fetch_embeddings = async |texts: &[&str]| {
        let e = gemini::fetch_embeddings(&gemini_api_key, texts).await?;
        Ok(e)
    };

    for arg in std::env::args() {
        let files = ds.add_directory(&arg).await?;
        for file in files {
            println!("Generating embeddings for {}", file.display());
            ds.embed_file(
                file.to_str()
                    .expect("path contains invalid utf-8 sequence")
                    .to_string(),
                fetch_embeddings,
            )
            .await?;
        }
    }

    println!("Initialized with {} nodes", ds.vs.header().nodes);

    run_cli_loop(&mut ds, &gemini_api_key).await?;

    Ok(())
}

async fn run_cli_loop<const D: usize>(
    ds: &mut DocumentStore<D>,
    gemini_api_key: &str,
) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = io::stdout();

    let fetch_embeddings = async |texts: &[&str]| {
        let e = gemini::fetch_embeddings(gemini_api_key, texts).await?;
        Ok(e)
    };

    loop {
        stdout.write_all(b"> ").await?;
        stdout.flush().await?;

        let mut line = String::new();
        tokio::select! {
            res = reader.read_line(&mut line) => {
                match res {
                    Ok(0) => break,
                    Ok(_) => {
                        let query_text = line.trim();
                        if query_text.is_empty() {
                            continue;
                        }
                        match gemini::fetch_embeddings(&gemini_api_key, &[query_text]).await {
                            Ok(query_vec) => {
                                if let Some(first_vec) = query_vec.into_iter().next() {
                                    let results = ds.search(&first_vec, 3).await?;
                                    for f in results {
                                        println!("{}", f.0);
                                    }
                                }
                            }
                            Err(e) => println!("API error: {}", e),
                        }
                    }
                    Err(e) => println!("line read error: {}", e),
                }
            }
            ev_opt = ds.next_event() => {
                if let Some(ev) = ev_opt {
                    println!("Event: {:?} [{}]", ev.etype, ev.file_path);
                    match ev.etype {
                        EventType::Create => {
                            if let Err(e) = ds.embed_file(ev.file_path, fetch_embeddings).await {
                                println!("Failed to embed file: {}", e);
                            }
                        }
                        EventType::Modify => {
                            if let Err(e) = ds.delete_file(&ev.file_path).await {
                                println!("Failed to delete old embeddings: {}", e);
                            }
                            if let Err(e) = ds.embed_file(ev.file_path, fetch_embeddings).await {
                                println!("Failed to embed updated file: {}", e);
                            }
                        }
                        EventType::Delete => {
                            if let Err(e) = ds.delete_file(&ev.file_path).await {
                                println!("Failed to delete embeddings: {}", e);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
