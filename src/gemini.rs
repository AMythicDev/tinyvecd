use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct EmbeddingRequestBody<'a> {
    pub model: &'a str,
    #[serde(rename = "taskType")]
    pub task_type: &'a str,
    pub content: Content<'a>,
}

#[derive(Serialize)]
pub struct Content<'a> {
    pub parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
pub struct Part<'a> {
    pub text: &'a str,
}

#[derive(Serialize)]
pub struct EmbeddingRequest<'a> {
    pub requests: Vec<EmbeddingRequestBody<'a>>,
}

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Embedding>,
}

#[derive(Deserialize)]
pub struct Embedding {
    pub values: Vec<f32>,
}

pub async fn fetch_embeddings(api_key: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let client = Client::new();
    let uri = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents";

    let requests: Vec<EmbeddingRequestBody> = texts
        .iter()
        .map(|text| EmbeddingRequestBody {
            model: "models/text-embedding-001",
            task_type: "RETRIEVAL_DOCUMENT",
            content: Content {
                parts: vec![Part { text }],
            },
        })
        .collect();

    let request_body = EmbeddingRequest { requests };

    let response = client
        .post(uri)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?;

    if !response.status().is_success() {
        let err_text = response.text().await?;
        eprintln!("Gemini API Error: {}", err_text);
        return Err(anyhow::anyhow!("GeminiApiError"));
    }

    let response_body: EmbeddingResponse = response.json().await?;

    let results = response_body
        .embeddings
        .into_iter()
        .map(|emb| emb.values)
        .collect();

    Ok(results)
}
