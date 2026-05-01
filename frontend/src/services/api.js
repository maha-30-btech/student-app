import axios from 'axios';
const API = axios.create({ 
  baseURL: process.env.REACT_APP_API_URL 
    ? `${process.env.REACT_APP_API_URL}/api` 
    : '/api' 
});
export const getModels      = ()      => API.get('/models');
export const getModel       = (id)    => API.get(`/models/${id}`);
export const getFeatures    = (id)    => API.get(`/models/${id}/features`);
export const predict        = (body)  => API.post('/predictions', body);
export const getHistory     = (p)     => API.get('/history', { params: p });
export const deleteRecord   = (id)    => API.delete(`/history/${id}`);
export const getStats       = ()      => API.get('/stats');
export const getComparisons = ()      => API.get('/comparison');
export const getComparison  = (id)    => API.get(`/comparison/${id}`);
