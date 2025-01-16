import { supabase } from '../config/supabase.js';
import { Document } from '../models/document.js';
import AppError from '../utils/appError.js';
import catchAsync from '../utils/catchAsync.js';
import axios from 'axios';
import FormData from 'form-data';
import dotenv from 'dotenv';
dotenv.config();

// Send file to FastAPI for processing
export const sendFileToFastAPI = async (fileBuffer, fileName) => {
  const form = new FormData();
  form.append('file', fileBuffer, fileName);

  try {
    const response = await axios.post(process.env.PYTHON_UPLOAD, form, {
      headers: {
        ...form.getHeaders(),
      },
    });
    return response.data;
  } catch (error) {
    throw new Error('Failed to upload file');
  }
};

export const searchInDocuments = async (query, parsedData) => {
  try {
    const response = await axios.post(
      process.env.PYTHON_SEARCH,
      { query, parsedData },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    return {
      results: response?.data?.results,
      totalResults: response?.data?.totalResults,
      query: response?.data?.query,
    };
  } catch (error) {
    throw new Error('Failed to query documents');
  }
};

// Search for documents using stored embeddings
export const searchDocuments = async (req, res) => {
  try {
    const { query, documentIds } = req.body;

    if (!query || !documentIds || !Array.isArray(documentIds)) {
      return res
        .status(400)
        .json({ error: 'Query and valid document IDs are required' });
    }

    const documents = await Document.find({ _id: { $in: documentIds } });

    if (!documents.length) {
      return res
        .status(404)
        .json({ error: 'No documents found for the provided IDs' });
    }

    // Include text content along with embeddings
    const parsedData = documents.map((doc) => ({
      embedding: doc.embedding,
      text: doc.parsedText,
      metadata: {
        name: doc.name,
        id: doc._id,
        fileType: doc.fileType,
        uploadDate: doc.uploadDate,
      },
    }));

    // Perform search in FastAPI with embeddings and text content
    const searchResponse = await searchInDocuments(query, parsedData);

    res.json({
      status: 'success',
      data: {
        results: searchResponse.results?.map((result) => ({
          ...result,
          matches: result.matches.map((match) => ({
            page: match.page_number || 'N/A',
            score: match?.similarity || 0,
            content: match.text || 'No Content Found',
            metadata: {
              ...match.metadata,
              similarity: match.similarity,
            },
          })),
        })),
        totalResults: searchResponse?.totalResults || 0,
        query: searchResponse?.query,
      },
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: 'Failed to search documents',
      message: error.message,
    });
  }
};

export const uploadFile = catchAsync(async (req, res, next) => {
  const { file } = req;

  if (!file) {
    return next(new AppError('No file uploaded', 400));
  }

  const { data, error } = await supabase.storage
    .from('Aethemis')
    .upload(`${Date.now()}-${file.originalname}`, file.buffer, {
      contentType: file.mimetype,
    });

  if (error) {
    return next(new AppError('Failed to upload file to Supabase', 500));
  }

  const publicUrl = supabase.storage.from('Aethemis').getPublicUrl(data.path)
    .data.publicUrl;

  const uploadedDoc = await sendFileToFastAPI(file.buffer, file.originalname);
  const parsedText = uploadedDoc.parsed_text;
  const embedding = uploadedDoc.embeddings;
  // Get the embeddings from FastAPI response

  // Save document metadata, parsed text, and embeddings to MongoDB
  const document = new Document({
    name: file.originalname,
    type: file.mimetype,
    size: file.size,
    url: publicUrl,
    path: data.path,
    parsedText: parsedText,
    embedding: embedding,
  });

  await document.save();

  res.status(201).json({
    status: 'success',
    data: {
      id: document._id,
      name: document.name,
      type: document.type,
      size: document.size,
      url: publicUrl,
      parsedText: document.parsedText,
    },
  });
});

export const getDocuments = catchAsync(async (req, res, next) => {
  const { ids } = req.body;

  if (!ids || !Array.isArray(ids)) {
    return res.status(400).json({ message: 'Invalid or missing document IDs' });
  }

  // Find documents matching the provided IDs
  const documents = await Document.find({ _id: { $in: ids } }, { content: 0 });
  res.json(documents);
});

export const deleteDocument = catchAsync(async (req, res, next) => {
  const document = await Document.findById(req.params.id);

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
