  import axios from 'axios';
  import FormData from 'form-data';

  // Send file to FastAPI for processing
  export const sendFileToFastAPI = async (fileBuffer, fileName) => {
    const form = new FormData();
    form.append('file', fileBuffer, fileName);

    try {
      const response = await axios.post('http://localhost:8080/upload', form, {
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

  // Send query to FastAPI for searching
  export const searchInDocuments = async (query) => {
    try {
      const response = await axios.post('http://localhost:8080/search', {
        query,
      });
      return response.data;
    } catch (error) {
      console.error('Error querying documents in FastAPI:', error);
      throw new Error('Failed to query documents');
    }
  };
