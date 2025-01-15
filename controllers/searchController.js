// services/searchService.js
import axios from 'axios';
import { Document } from '../models/document.js';

// Search using MongoDB
export const searchDocuments = async (query) => {
  const searchRegex = new RegExp(query, 'i');

  // Search in MongoDB
  const localResults = await Document.find({
    $or: [
      { name: searchRegex },
      { type: searchRegex },
      { content: searchRegex },
    ],
  });

  return localResults;
};

export const searchDocumentsWithFastAPI = async (query) => {
  const apiUrl = process.env.FASTAPI_URL;

  const response = await axios.post(`${apiUrl}/search`, { query });

  if (response.status !== 200) {
    throw new Error('Error fetching results from FastAPI');
  }

  return response.data.results;
};
