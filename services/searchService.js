import fetch from 'node-fetch';

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Search documents using FastAPI service
export async function searchDocuments(query) {
  try {
    const response = await fetch(`${FASTAPI_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      throw new Error('Search request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Search service error:', error);
    throw error;
  }
}