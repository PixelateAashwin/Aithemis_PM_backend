// app.js
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import { fileRouter } from './routes/files.js';
import { searchRouter } from './routes/search.js';
import AppError from './utils/appError.js';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// Routes
app.use('/api/files', fileRouter);
app.use('/api/search', searchRouter);

// Handle 404
app.all('*', (req, res, next) => {
  next(new AppError(`Cannot find ${req.originalUrl} on this server`, 404));
});

// Global error handler
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
  console.log(err);
  res.status(statusCode).json({
    status,
    message: err.message,
  });
});

export default app;
