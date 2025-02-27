import mongoose from 'mongoose';

const documentSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  type: {
    type: String,
    required: true,
  },
  size: {
    type: Number,
    required: true,
  },
  path: {
    type: String,
    required: true,
  },
  parsedText: {
    type: String,
  },
  url: {
    type: String,
    required: true,
  },
  embedding: {
    type: [[Number]],
  },
  uploadedAt: {
    type: Date,
    default: Date.now(),
  },
});

export const Document = mongoose.model('Document', documentSchema);
