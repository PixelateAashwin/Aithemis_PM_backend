// routes/search.js
import express from 'express';
import * as fileController from '../controllers/fileController.js';
const router = express.Router();

// Search endpoint
router.post('/', fileController.searchDocuments);
export const searchRouter = router;
