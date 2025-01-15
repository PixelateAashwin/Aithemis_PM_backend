// routes/files.js
import express from 'express';
import multer from 'multer';
import * as fileController from '../controllers/fileController.js';

const router = express.Router();
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Routes
router.post('/upload', upload.single('file'), fileController.uploadFile);
router.post('/', fileController.getDocuments);
router.delete('/:id', fileController.deleteDocument);

export const fileRouter = router;
