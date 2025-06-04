// La începutul fișierului, îi spunem Next.js că această componentă
// va interacționa cu utilizatorul în browser.
'use client';

// Importăm funcționalități de la React care ne ajută să gestionăm starea
// și evenimentele (cum ar fi click-ul pe buton).
import { useState, FormEvent } from 'react';
// Importăm axios DUPĂ ce l-ai instalat cu 'npm install axios'
import axios from 'axios';

// Definim cum arată un document sursă pe care îl vom primi de la API
interface SourceDoc {
  page_content: string; // Conținutul fragmentului de text
  metadata: {
    source?: string; // Numele fișierului sursă (ex: "cod_rutier.pdf")
    // Aici pot fi și alte metadate, de ex: page_number
    [key: string]: any;
  };
  score?: number | null; // Scorul de similaritate
}

// Definim cum arată răspunsul complet de la API-ul nostru Python
interface ApiResponse {
  answer: string;                 // Răspunsul generat de LLM
  source_documents: SourceDoc[];  // Lista documentelor sursă
  processing_time: number;        // Timpul cât a durat procesarea pe backend
}

// Aceasta este componenta principală a paginii tale
export default function HomePage() {
  // Aici definim "stările" componentei noastre folosind hook-ul useState
  const [question, setQuestion] = useState<string>('');
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); 

    if (!question.trim()) {
      setError('Te rog scrie o întrebare.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setApiResponse(null);

    try {
      const response = await axios.post<ApiResponse>(
        'http://localhost:8000/ask', 
        { question: question }
      );
      setApiResponse(response.data);
    } catch (err) {
      console.error("Eroare la apelul API:", err);
      if (axios.isAxiosError(err) && err.response) {
        setError(`Eroare de la server: ${err.response.data.detail || err.message}`);
      } else if (err instanceof Error) { // Verifică dacă err este o instanță a clasei Error
        setError(`Nu am putut contacta serverul sau a apărut o altă eroare: ${err.message}`);
      } else {
        setError('A apărut o eroare necunoscută.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '700px', margin: '30px auto', padding: '20px', fontFamily: 'sans-serif', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>Asistent Cod Rutier (RAG)</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Introdu întrebarea ta aici..."
          rows={5}
          disabled={isLoading}
          style={{ width: '100%', padding: '10px', marginBottom: '10px', border: '1px solid #ccc', borderRadius: '4px', boxSizing: 'border-box' }}
        />
        <button
          type="submit"
          disabled={isLoading}
          style={{
            display: 'block',
            width: '100%',
            padding: '12px',
            backgroundColor: isLoading ? '#aaa' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {isLoading ? 'Se procesează...' : 'Trimite Întrebarea'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '20px', padding: '10px', color: 'red', backgroundColor: '#ffebee', border: '1px solid red', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {isLoading && (
        <div style={{ marginTop: '20px', textAlign: 'center', color: '#555' }}>
          <p>Se încarcă răspunsul... Poate dura câteva momente.</p>
        </div>
      )}

      {apiResponse && !isLoading && (
        <div style={{ marginTop: '30px', borderTop: '1px solid #eee', paddingTop: '20px' }}>
          <h2 style={{ color: '#333' }}>Răspuns:</h2>
          <div style={{ padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>
            {apiResponse.answer}
          </div>
          <p style={{ fontSize: '0.9em', color: '#777', marginTop: '10px' }}>
            Timp procesare backend: {apiResponse.processing_time.toFixed(2)} secunde.
          </p>

          {apiResponse.source_documents && apiResponse.source_documents.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h3 style={{ color: '#333' }}>Surse folosite:</h3>
              {apiResponse.source_documents.map((doc, index) => (
                <div key={index} style={{ border: '1px solid #e0e0e0', padding: '10px', marginBottom: '10px', borderRadius: '4px', backgroundColor: '#fff' }}>
                  <p style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                    Sursă {index + 1}: {doc.metadata?.source || 'Necunoscută'}
                    {doc.score !== undefined && doc.score !== null && (
                      <span style={{ marginLeft: '10px', color: '#555', fontSize: '0.9em', fontWeight: 'normal' }}>(Scor: {doc.score.toFixed(4)})</span>
                    )}
                  </p>
                  <p style={{ maxHeight: '100px', overflowY: 'auto', fontSize: '0.9em', backgroundColor: '#fdfdfd', padding: '8px', border: '1px solid #eee', borderRadius: '3px', whiteSpace: 'pre-wrap' }}>
                    <em>"{doc.page_content.substring(0, 300)}..."</em>
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