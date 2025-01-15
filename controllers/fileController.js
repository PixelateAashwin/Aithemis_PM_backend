import { supabase } from '../config/supabase.js';
import { Document } from '../models/document.js';
import AppError from '../utils/appError.js';
import catchAsync from '../utils/catchAsync.js';
import axios from 'axios';
import FormData from 'form-data';

// Send file to FastAPI for processing
export const sendFileToFastAPI = async (fileBuffer, fileName) => {
  const form = new FormData();
  form.append('file', fileBuffer, fileName);

  try {
    const response = await axios.post('https://aithemis-pm-backend-1.onrender.com/upload', form, {
      headers: {
        ...form.getHeaders(),
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading file to FastAPI:', error);
    throw new Error('Failed to upload file');
  }
};

// Updated function to search documents in FastAPI
export const searchInDocuments = async (query) => {
  try {
    const response = await axios.post(
      'https://aithemis-pm-backend-1.onrender.com/search',
      { query: query },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    // Return the complete results object
    return {
      results: response.data.results,
      totalResults: response.data.total_results,
      query: response.data.query,
    };
  } catch (error) {
    console.error('Error querying documents in FastAPI:', error);
    throw new Error('Failed to query documents');
  }
};

// Updated controller function to handle the search query
export const searchDocuments = async (req, res) => {
  try {
    const { query } = req.body;
    console.log('Received search query:', query);

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Get search results from FastAPI
    const searchResponse = await searchInDocuments(query);

    // Return formatted response
    return res.json({
      status: 'success',
      data: {
        results: searchResponse.results.map((result) => ({
          content: result.content,
          metadata: result.metadata,
          score: result.score,
          snippet: result.snippet,
        })),
        totalResults: searchResponse.totalResults,
        query: searchResponse.query,
      },
    });
  } catch (error) {
    console.error('Error querying documents:', error.message);
    return res.status(500).json({
      status: 'error',
      error: 'Failed to query documents',
      message: error.message,
    });
  }
};

export const uploadFile = catchAsync(async (req, res, next) => {
  const { file } = req;
  console.log(file);

  if (!file) {
    return next(new AppError('No file uploaded', 400));
  }

  const { data, error } = await supabase.storage
    .from('Aethemis')
    .upload(`${Date.now()}-${file.originalname}`, file.buffer, {
      contentType: file.mimetype,
    });

  if (error) {
    console.log(error);
    return next(new AppError('Failed to upload file to Supabase', 500));
  }

  const publicUrl = supabase.storage.from('Aethemis').getPublicUrl(data.path)
    .data.publicUrl;

  // Save document metadata in MongoDB
  const document = new Document({
    name: file.originalname,
    type: file.mimetype,
    size: file.size,
    url: publicUrl,
    path: data.path,
  });
  console.log(document);

  await document.save();

  // Send the file to FastAPI for processing
  await sendFileToFastAPI(file.buffer, file.originalname);

  res.status(201).json({
    status: 'success',
    data: {
      id: document._id,
      name: document.name,
      type: document.type,
      size: document.size,
      url: publicUrl,
    },
  });
});

// Fetch all documents
export const getDocuments = catchAsync(async (req, res, next) => {
  const { ids } = req.body; // Expect an array of document IDs in the request body

  if (!ids || !Array.isArray(ids)) {
    return res.status(400).json({ message: 'Invalid or missing document IDs' });
  }

  // Find documents matching the provided IDs
  const documents = await Document.find({ _id: { $in: ids } }, { content: 0 });
  res.json(documents);
});

// Delete document
export const deleteDocument = catchAsync(async (req, res, next) => {
  const document = await Document.findById(req.params.id);
  console.log(document);

  if (!document) {
    return next(new AppError('Document not found', 404));
  }

  const { data, error } = await supabase.storage
    .from('Aethemis')
    .remove([document.path]);

  if (error) {
    return next(new AppError('Failed to delete file from Supabase', 500));
  }

  await document.deleteOne();
  res.status(204).send();
});
