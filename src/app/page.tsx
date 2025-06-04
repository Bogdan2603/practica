'use client';

import { useState, FormEvent } from 'react';
import axios from 'axios';

interface SourceDoc {
  page_content: string;
  metadata: {
    source?: string;
    [key: string]: any;
  };
  score?: number | null;
}

interface ApiResponse {
  answer: string;
  source_documents: SourceDoc[];
  processing_time: number;
}

export default function HomePage() {
  const [question, setQuestion] = useState<string>('');
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim()) {
      setError('Întrebarea nu poate fi goală.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setApiResponse(null);
    try {
      const response = await axios.post<ApiResponse>(
        'http://localhost:8000/ask',
        { question }
      );
      setApiResponse(response.data);
    } catch (err) {
      console.error("API Error:", err);
      if (axios.isAxiosError(err) && err.response) {
        setError(`Eroare API: ${err.response.data.detail || err.message}`);
      } else if (err instanceof Error) {
        setError(`A apărut o eroare la procesarea întrebării: ${err.message}`);
      } else {
        setError('A apărut o eroare necunoscută.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Definim câteva culori și fonturi pentru un aspect mai modern
  const colors = {
    primary: '#007bff', // Un albastru vibrant
    primaryDark: '#0056b3',
    lightBg: '#f8f9fa',
    darkText: '#212529',
    mediumText: '#495057',
    lightText: '#6c757d',
    border: '#dee2e6',
    errorRed: '#dc3545',
    errorBg: '#f8d7da',
  };

  const fonts = {
    main: '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  };

  return (
    <div style={{
      fontFamily: fonts.main,
      maxWidth: '800px',
      margin: '40px auto',
      padding: '30px',
      backgroundColor: '#fff', // Fundal alb pentru containerul principal
      borderRadius: '12px',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)', // Umbră mai pronunțată
    }}>
      <header style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ color: colors.darkText, fontSize: '2.25rem', fontWeight: 600 }}>
          Asistent Cod Rutier <span style={{color: colors.primary, fontWeight: 700 }}>(RAG)</span>
        </h1>
      </header>

      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Scrie întrebarea ta aici..."
          rows={5}
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '15px',
            marginBottom: '15px',
            border: `1px solid ${colors.border}`,
            borderRadius: '8px',
            fontSize: '1rem',
            lineHeight: '1.5',
            boxSizing: 'border-box',
            resize: 'vertical',
            transition: 'border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            // Efect la focus
            // ':focus': { borderColor: colors.primary, boxShadow: `0 0 0 0.2rem rgba(0,123,255,.25)` } // Nu funcționează în stiluri inline
          }}
        />
        <button
          type="submit"
          disabled={isLoading}
          style={{
            display: 'block',
            width: '100%',
            padding: '12px 20px',
            backgroundColor: isLoading ? colors.mediumText : colors.primary,
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '1.1rem',
            fontWeight: 600,
            cursor: isLoading ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s ease-in-out',
            // Efect la hover (nu funcționează în stiluri inline așa)
            // ':hover': { backgroundColor: isLoading ? colors.mediumText : colors.primaryDark }
          }}
        >
          {isLoading ? 'Se procesează...' : 'Trimite Întrebarea'}
        </button>
      </form>

      {error && (
        <div style={{
          marginTop: '20px',
          padding: '15px',
          color: colors.errorRed,
          backgroundColor: colors.errorBg,
          border: `1px solid ${colors.errorRed}`,
          borderRadius: '8px',
          textAlign: 'center',
        }}>
          {error}
        </div>
      )}

      {isLoading && (
        <div style={{ marginTop: '25px', textAlign: 'center', color: colors.lightText, padding: '20px' }}>
          <p style={{fontSize: '1.1rem'}}>Se încarcă răspunsul...</p>
          {/* Aici ai putea adăuga un spinner SVG sau o animație CSS */}
        </div>
      )}

      {apiResponse && !isLoading && (
        <div style={{ marginTop: '30px', borderTop: `1px solid ${colors.border}`, paddingTop: '25px' }}>
          <div>
            <h2 style={{ color: colors.darkText, fontSize: '1.75rem', marginBottom: '15px' }}>Răspuns:</h2>
            <div style={{
              padding: '20px',
              backgroundColor: colors.lightBg,
              borderRadius: '8px',
              marginBottom: '10px',
              whiteSpace: 'pre-wrap', // Păstrează formatarea textului (spații, linii noi)
              lineHeight: '1.6',
              color: colors.mediumText,
              border: `1px solid ${colors.border}`
            }}>
              {apiResponse.answer}
            </div>
            <p style={{ fontSize: '0.9em', color: colors.lightText, textAlign: 'right' }}>
              Timp procesare backend: {apiResponse.processing_time.toFixed(2)} secunde.
            </p>
          </div>

          {apiResponse.source_documents && apiResponse.source_documents.length > 0 && (
            <div style={{ marginTop: '30px' }}>
              <h3 style={{ color: colors.darkText, fontSize: '1.5rem', marginBottom: '15px' }}>Surse Folosite:</h3>
              {apiResponse.source_documents.map((doc, index) => (
                <div key={index} style={{
                  border: `1px solid ${colors.border}`,
                  padding: '15px',
                  marginBottom: '15px',
                  borderRadius: '8px',
                  backgroundColor: '#fff', // Fundal alb pentru cardurile sursă
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)'
                }}>
                  <p style={{ fontWeight: 'bold', color: colors.mediumText, marginBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Sursa {index + 1}: {doc.metadata?.source || 'Necunoscută'}</span>
                    {doc.score !== undefined && doc.score !== null && (
                      <span style={{
                        backgroundColor: doc.score > 0.5 ? '#28a745' : (doc.score > 0.3 ? '#ffc107' : '#dc3545'), // Verde, Galben, Roșu
                        color: 'white',
                        padding: '3px 8px',
                        borderRadius: '12px',
                        fontSize: '0.8em',
                        fontWeight: 'normal'
                      }}>
                        Scor: {doc.score.toFixed(3)}
                      </span>
                    )}
                  </p>
                  <p style={{
                    maxHeight: '120px',
                    overflowY: 'auto',
                    fontSize: '0.95em',
                    backgroundColor: colors.lightBg,
                    padding: '10px',
                    border: `1px solid #e9ecef`,
                    borderRadius: '6px',
                    whiteSpace: 'pre-wrap',
                    lineHeight: '1.5',
                    color: colors.lightText
                  }}>
                    <em>"{doc.page_content.substring(0, 350)}..."</em>
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}