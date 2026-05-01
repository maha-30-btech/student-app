import React, { useEffect, useState } from 'react';
import { getHistory, deleteRecord } from '../services/api';

export default function History() {
  const [records, setRecords] = useState([]);
  const [total,   setTotal]   = useState(0);
  const [page,    setPage]    = useState(1);
  const [pages,   setPages]   = useState(1);
  const [loading, setLoading] = useState(true);
  const [filter,  setFilter]  = useState({ dataset: '', target: '' });

  const load = (p = 1) => {
    setLoading(true);
    const params = { page: p, limit: 15, ...filter };
    Object.keys(params).forEach(k => { if (!params[k]) delete params[k]; });
    getHistory(params).then(r => {
      setRecords(r.data.records || []);
      setTotal(r.data.total || 0);
      setPages(r.data.pages || 1);
      setPage(p);
      setLoading(false);
    });
  };

  useEffect(() => { load(1); }, [filter]);

  const handleDelete = async (id) => {
    await deleteRecord(id);
    load(page);
  };

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-title">Prediction History</div>
        <div className="page-sub">{total} predictions stored in MongoDB</div>
      </div>

      {/* Filters */}
      <div className="card" style={{ marginBottom: 20, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, color: 'var(--text2)' }}>Filter:</span>
        <select value={filter.dataset} onChange={e => setFilter(f => ({ ...f, dataset: e.target.value }))} style={{ width: 150 }}>
          <option value="">All Datasets</option>
          <option value="math">Mathematics</option>
          <option value="port">Portuguese</option>
          <option value="exam">Exam</option>
        </select>
        <select value={filter.target} onChange={e => setFilter(f => ({ ...f, target: e.target.value }))} style={{ width: 150 }}>
          <option value="">All Targets</option>
          {['G1','G2','G3','math_score','reading_score','writing_score'].map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button className="btn btn-ghost" style={{ fontSize: 13, padding: '7px 12px' }}
          onClick={() => setFilter({ dataset:'', target:'' })}>
          ✕ Clear
        </button>
      </div>

      {loading ? (
        <div style={{ display:'flex', gap:10, color:'var(--text2)', padding:24 }}><div className="spinner"/>Loading...</div>
      ) : records.length === 0 ? (
        <div className="card" style={{ textAlign:'center', padding:48 }}>
          <div style={{ fontSize:36, opacity:0.2, marginBottom:12 }}>▦</div>
          <div style={{ color:'var(--text2)' }}>No predictions yet. Try making one!</div>
        </div>
      ) : (
        <>
          <div className="card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Dataset</th>
                  <th>Target</th>
                  <th>Prediction</th>
                  <th>Top Features</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {records.map(r => {
                  const isScore = r.target?.includes('score');
                  const max  = isScore ? 100 : 20;
                  const pct  = (r.prediction / max) * 100;
                  const col  = pct > 70 ? 'var(--accent)' : pct > 40 ? 'var(--accent4)' : 'var(--accent3)';
                  return (
                    <tr key={r._id}>
                      <td style={{ fontSize:12, color:'var(--text3)' }}>
                        {new Date(r.timestamp).toLocaleString()}
                      </td>
                      <td><span className="badge badge-blue">{r.dataset}</span></td>
                      <td style={{ fontFamily:'var(--mono)', fontSize:12 }}>{r.target}</td>
                      <td>
                        <span style={{ fontFamily:'var(--mono)', fontWeight:700, color: col, fontSize:15 }}>
                          {r.prediction}
                        </span>
                        <span style={{ color:'var(--text3)', fontSize:12 }}>/{max}</span>
                      </td>
                      <td style={{ fontSize:11, color:'var(--text3)' }}>
                        {Object.entries(r.inputFeatures || {}).slice(0,3).map(([k,v]) =>
                          <span key={k} className="tag" style={{ marginRight:3 }}>{k}:{v}</span>
                        )}
                        {Object.keys(r.inputFeatures || {}).length > 3 && (
                          <span style={{ color:'var(--text3)', fontSize:10 }}>+{Object.keys(r.inputFeatures).length-3} more</span>
                        )}
                      </td>
                      <td>
                        <button className="btn btn-danger" style={{ fontSize:11, padding:'4px 10px' }}
                          onClick={() => handleDelete(r._id)}>
                          Delete
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pages > 1 && (
            <div style={{ display:'flex', gap:8, justifyContent:'center', marginTop:16 }}>
              <button className="btn btn-ghost" disabled={page===1} onClick={() => load(page-1)} style={{ padding:'6px 14px' }}>← Prev</button>
              <span style={{ padding:'8px 12px', color:'var(--text2)', fontSize:13 }}>Page {page} of {pages}</span>
              <button className="btn btn-ghost" disabled={page===pages} onClick={() => load(page+1)} style={{ padding:'6px 14px' }}>Next →</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
