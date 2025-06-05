// src/app/chat/page.tsx
'use client';

import { useState, FormEvent, useEffect, useRef } from 'react';
import axios from 'axios';
import styles from './HomePage.module.css'; // Importul CSS Module
import { 
    FiSun, FiMoon, FiTrash2, FiSend, FiChevronDown, FiChevronUp, 
    FiMessageSquare, FiLoader, FiFileText, FiCopy, FiCheck 
} from 'react-icons/fi';

// --- Interfețe ---
interface SourceDoc {
  page_content: string;
  metadata: {
    source?: string;
    [key: string]: any;
  };
  score?: number | null;
}

interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot' | 'error';
  sources?: SourceDoc[];
  processingTime?: number;
  timestamp: Date; // Păstrăm ca Date
}

// Pentru parsarea din localStorage unde timestamp e string
interface ChatMessageFromStorage {
  id: string;
  text: string;
  sender: 'user' | 'bot' | 'error';
  sources?: SourceDoc[];
  processingTime?: number;
  timestamp: string; // Timestamp va fi string din JSON.parse
}

interface ApiResponse {
  answer: string;
  source_documents: SourceDoc[];
  processing_time: number;
}

const getThemeColors = (currentTheme: 'light' | 'dark') => {
  if (currentTheme === 'dark') {
    return {
      primary: '#58A6FF', primaryDark: '#388BF5', pageBackground: '#0D1117',
      cardBackground: '#161B22', textTitle: '#E6EDF3', textBody: '#C9D1D9',
      textMuted: '#768390', border: '#30363D', inputBackground: '#010409',
      userBubbleBg: '#238636', userBubbleText: '#FFFFFF', 
      botBubbleBg: '#21262D', botBubbleText: '#E6EDF3', 
      errorText: '#F85149', errorBubbleBg: '#491A1F',
      error: '#F85149', success: '#3FB950',
      scoreGoodBg: '#238636', scoreMediumBg: '#BD8B00', scoreBadBg: '#DA3633',
      exampleBtnBg: '#21262D', exampleBtnHoverBg: '#30363D', exampleBtnBorder: '#30363D',
      sourceSnippetBg: 'rgba(255,255,255,0.03)', sourceSnippetBorder: 'rgba(255,255,255,0.07)',
      userAvatarBg: '#238636', userAvatarText: '#FFFFFF',
      botAvatarBg: '#58A6FF', botAvatarText: '#0D1117',
    };
  }
  // Light Theme
  return {
    primary: '#4A90E2', primaryDark: '#357ABD', pageBackground: '#eef2f9',
    cardBackground: '#FFFFFF', textTitle: '#1A202C', textBody: '#4A5568',
    textMuted: '#718096', border: '#E2E8F0', inputBackground: '#F7FAFC',
    userBubbleBg: '#4A90E2', userBubbleText: '#FFFFFF', 
    botBubbleBg: '#e9e9eb', botBubbleText: '#1C1C1E', 
    errorText: '#D32F2F', errorBubbleBg: '#FFEBEE',
    error: '#D32F2F', success: '#38A169',
    scoreGoodBg: '#38A169', scoreMediumBg: '#DD6B20', scoreBadBg: '#D32F2F',
    exampleBtnBg: '#FFFFFF', exampleBtnHoverBg: '#F7FAFC', exampleBtnBorder: '#E2E8F0',
    sourceSnippetBg: '#F7FAFC', sourceSnippetBorder: '#E9ECEF',
    userAvatarBg: '#4A90E2', userAvatarText: '#FFFFFF',
    botAvatarBg: '#718096', botAvatarText: '#FFFFFF',
  };
};

export default function HomePage() {
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [expandedSourcesMap, setExpandedSourcesMap] = useState<Record<string, boolean>>({});
  const [copiedStatus, setCopiedStatus] = useState<Record<string, boolean>>({});
  const [initialLoadComplete, setInitialLoadComplete] = useState<boolean>(false); // Pentru a preveni salvarea la încărcarea inițială

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const currentColors = getThemeColors(theme);

  // Efect pentru scroll automat la mesaje noi
  useEffect(() => { 
    if (messages.length > 0) { // Scroll doar dacă sunt mesaje
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); 
    }
  }, [messages]);

  // Efect pentru încărcarea temei ȘI a mesajelor salvate la prima randare
  useEffect(() => { 
    const savedTheme = localStorage.getItem('chatTheme') as 'light' | 'dark'; 
    if (savedTheme) {
      setTheme(savedTheme); 
    }

    const savedMessagesString = localStorage.getItem('chatHistory');
    if (savedMessagesString) {
      try {
        const parsedMessagesFromStorage: ChatMessageFromStorage[] = JSON.parse(savedMessagesString);
        const hydratedMessages: ChatMessage[] = parsedMessagesFromStorage.map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp), // Rehidratează string-ul Date în obiect Date
        }));
        setMessages(hydratedMessages);
      } catch (e) {
        console.error("Nu s-a putut parsa istoricul chat-ului din localStorage:", e);
        localStorage.removeItem('chatHistory'); // Curăță istoricul corupt
      }
    }
    setInitialLoadComplete(true); // Marcăm că încărcarea inițială s-a finalizat
  }, []); // Array-ul gol asigură rularea o singură dată la montare

  // Efect pentru aplicarea temei și salvarea ei în localStorage când 'theme' se schimbă
  useEffect(() => {
    const colorsForEffect = getThemeColors(theme); 
    document.body.style.backgroundColor = colorsForEffect.pageBackground;
    if (theme === 'dark') {
      document.documentElement.classList.add('dark-theme-active');
    } else {
      document.documentElement.classList.remove('dark-theme-active');
    }
    localStorage.setItem('chatTheme', theme);
  }, [theme]); 

  // NOU: Efect pentru salvarea mesajelor în localStorage când 'messages' se schimbă
  useEffect(() => {
    if (initialLoadComplete) { // Salvează doar după ce încărcarea inițială s-a făcut
      localStorage.setItem('chatHistory', JSON.stringify(messages));
    }
  }, [messages, initialLoadComplete]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const questionText = currentQuestion.trim();
    if (!questionText) return;
    const userMessage: ChatMessage = { id: `user-${Date.now()}`, text: questionText, sender: 'user', timestamp: new Date() };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setCurrentQuestion('');
    setIsLoading(true);
    try {
      const apiResponse = await axios.post<ApiResponse>( 'http://localhost:8000/ask', { question: questionText });
      const botMessage: ChatMessage = { id: `bot-${Date.now()}`, text: apiResponse.data.answer, sender: 'bot', sources: apiResponse.data.source_documents, processingTime: apiResponse.data.processing_time, timestamp: new Date() };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (err) {
      console.error("API Error:", err);
      let errorMessageText = 'A apărut o eroare la procesarea întrebării.';
      if (axios.isAxiosError(err) && err.response) errorMessageText = `Eroare API: ${err.response.data.detail || err.message}`;
      else if (err instanceof Error) errorMessageText = err.message;
      const errorMessage: ChatMessage = { id: `error-${Date.now()}`, text: errorMessageText, sender: 'error', timestamp: new Date() };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const getScoreBadgeStyle = (score: number | null | undefined): React.CSSProperties => {
    let bgColor = currentColors.textMuted;
    if (score !== null && score !== undefined) {
        if (score > 0.55) bgColor = currentColors.scoreGoodBg;
        else if (score > 0.35) bgColor = currentColors.scoreMediumBg; 
        else bgColor = currentColors.scoreBadBg;
    }
    return { backgroundColor: bgColor, color: 'white', padding: '5px 10px', borderRadius: '9999px', fontSize: '0.7rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.4px', flexShrink: 0 };
  };

  const getMessageBubbleStyles = (sender: ChatMessage['sender']): React.CSSProperties => {
    const baseBubbleStyle: React.CSSProperties = { padding: '10px 15px', maxWidth: '100%', boxShadow: '0 2px 4px rgba(0,0,0,0.07)', wordBreak: 'break-word', lineHeight: '1.6', minWidth: '80px' };
    if (sender === 'user') return { ...baseBubbleStyle, backgroundColor: currentColors.userBubbleBg, color: currentColors.userBubbleText, borderRadius: '20px 20px 5px 20px' };
    if (sender === 'bot') return { ...baseBubbleStyle, backgroundColor: currentColors.botBubbleBg, color: currentColors.botBubbleText, borderRadius: '20px 20px 20px 5px' };
    return { ...baseBubbleStyle, backgroundColor: currentColors.errorBubbleBg, color: currentColors.errorText, borderRadius: '10px' };
  };

  const AvatarComponent = ({ sender }: { sender: ChatMessage['sender'] }) => {
    const avatarSize = '32px'; let initial = '';
    let avatarStyle: React.CSSProperties = { width: avatarSize, height: avatarSize, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.9rem', fontWeight: 'bold', flexShrink: 0, marginTop: '4px' };
    if (sender === 'user') { initial = 'U'; avatarStyle.backgroundColor = currentColors.userAvatarBg; avatarStyle.color = currentColors.userAvatarText; }
    else if (sender === 'bot') { initial = 'A'; avatarStyle.backgroundColor = currentColors.botAvatarBg; avatarStyle.color = currentColors.botAvatarText; }
    else { initial = '!'; avatarStyle.backgroundColor = currentColors.errorBubbleBg; avatarStyle.color = currentColors.errorText; }
    return <div style={avatarStyle}>{initial}</div>;
  };

  const toggleTheme = () => setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  
  // Actualizăm handleClearChat pentru a șterge și din localStorage
  const handleClearChat = () => {
    setMessages([]);
    localStorage.removeItem('chatHistory'); // Șterge istoricul din localStorage
  };

  const exampleQuestions = [ "Ce este un autovehicul?", "Care este limita de viteză în localitate?", "Ce documente trebuie să am la mine când conduc?", "Explică prioritatea de dreapta."];
  const handleExampleQuestionClick = (question: string) => { setCurrentQuestion(question); };
  const toggleSourceExpansion = (messageId: string) => setExpandedSourcesMap(prev => ({ ...prev, [messageId]: !prev[messageId] }));
  const handleSourceClick = (source: SourceDoc) => {
    console.log("Sursă selectată:", source);
    alert(`Ați selectat sursa: ${source.metadata?.source || 'Necunoscută'}\nFragment: "${source.page_content.substring(0,150)}..."`);
  };
  const handleCopyText = (textToCopy: string, messageId: string) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(textToCopy)
        .then(() => {
          setCopiedStatus(prev => ({ ...prev, [messageId]: true }));
          setTimeout(() => { setCopiedStatus(prev => ({ ...prev, [messageId]: false })); }, 2000);
        })
        .catch(err => console.error('Nu s-a putut copia textul: ', err));
    } else { console.warn('API-ul Clipboard nu este disponibil în acest browser.'); }
  };

  return (
    <div className={styles.pageContainer} style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: currentColors.pageBackground, color: currentColors.textBody }}>
      <header style={{
          flexShrink: 0, backgroundColor: currentColors.cardBackground, padding: '10px 15px',
          borderBottom: `1px solid ${currentColors.border}`, display: 'flex',
          justifyContent: 'space-between', alignItems: 'center', minHeight: '50px',
      }}>
        <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
          <FiMessageSquare size={20} style={{color: currentColors.primary, flexShrink: 0}}/>
          <div>
            <h1 style={{fontSize: '1.05rem', lineHeight: '1.2', margin: 0, color: currentColors.textTitle, fontWeight: 600 }}>Asistent Rutier</h1>
            <p style={{fontSize: '0.65rem', lineHeight: '1.2', margin: '2px 0 0 0', color: currentColors.textMuted}}>RAG & AI</p>
          </div>
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            <button onClick={handleClearChat} title="Șterge conversația" style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '5px', color: currentColors.textMuted, display: 'flex', alignItems: 'center' }} onMouseEnter={(e) => e.currentTarget.style.color = currentColors.primary} onMouseLeave={(e) => e.currentTarget.style.color = currentColors.textMuted}>
                <FiTrash2 size={18} /> <span style={{marginLeft:'4px', fontSize:'0.8rem'}}>Șterge</span>
            </button>
            <button onClick={toggleTheme} title="Comută tema" style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '5px', color: currentColors.textMuted, display: 'flex', alignItems: 'center' }} onMouseEnter={(e) => e.currentTarget.style.color = currentColors.primary} onMouseLeave={(e) => e.currentTarget.style.color = currentColors.textMuted}>
                {theme === 'light' ? <FiMoon size={18} /> : <FiSun size={18} />}
                <span style={{marginLeft: '4px', fontSize:'0.8rem'}}>{theme === 'light' ? 'Întunecat' : 'Luminos'}</span>
            </button>
        </div>
      </header>

      <div style={{ 
          flexGrow: 1, overflowY: 'auto', padding: '20px 15px', 
          display: 'flex', flexDirection: 'column', gap: '18px'
      }}>
        {messages.length === 0 && !isLoading && (
          <div className={styles.messageAppear} style={{ textAlign: 'center', padding: '20px', color: currentColors.textMuted, flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
            <FiMessageSquare size={40} style={{marginBottom: '15px', color: currentColors.textMuted+'99'}} />
            <h2 style={{ fontSize: '1.3rem', color: currentColors.textTitle, marginBottom: '8px', fontWeight:600 }}>Bine ai venit!</h2>
            <p style={{fontSize: '0.85rem', marginBottom: '20px', maxWidth: '350px' }}> Sunt asistentul tău pentru Codul Rutier. Încearcă una din întrebările de mai jos sau scrie una proprie! </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%', maxWidth: '350px' }}>
              {exampleQuestions.map((q, i) => (
                <button key={i} onClick={() => handleExampleQuestionClick(q)} style={{ padding: '10px 15px', backgroundColor: currentColors.cardBackground, border: `1px solid ${currentColors.border}`, borderRadius: '8px', cursor: 'pointer', textAlign: 'left', color: currentColors.textBody, boxShadow: '0 1px 2px rgba(0,0,0,0.04)', transition: 'background-color 0.2s, transform 0.1s', fontSize:'0.85rem' }} onMouseEnter={(e) => {e.currentTarget.style.backgroundColor = currentColors.inputBackground; e.currentTarget.style.transform = 'scale(1.01)';}} onMouseLeave={(e) => {e.currentTarget.style.backgroundColor = currentColors.cardBackground; e.currentTarget.style.transform = 'scale(1)';}} >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => {
          const bubbleStylesWithSender = getMessageBubbleStyles(msg.sender);
          const isUser = msg.sender === 'user';
          const formattedTimestamp = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          const botAvatar = !isUser ? <AvatarComponent sender={msg.sender} /> : null;
          const userAvatar = isUser ? <AvatarComponent sender={msg.sender} /> : null;

          return (
            <div key={msg.id} className={styles.messageAppear} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
                {botAvatar}
                <div style={{ maxWidth: '75%', display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start' }}>
                    <div style={{...bubbleStylesWithSender, position: 'relative' }}>
                        <p style={{ margin: 0, whiteSpace: 'pre-wrap', paddingRight: msg.sender === 'bot' ? '30px' : '0' }}>{msg.text}</p>
                        {msg.sender === 'bot' && !isLoading && (
                            <button onClick={() => handleCopyText(msg.text, msg.id)} title="Copiază textul" style={{ position: 'absolute', top: '4px', right: '4px', background: 'rgba(0,0,0,0.1)', border: 'none', borderRadius: '50%', padding: '4px', cursor: 'pointer', color: bubbleStylesWithSender.color as string, opacity: 0.6, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onMouseEnter={(e) => e.currentTarget.style.opacity = '1'} onMouseLeave={(e) => { if (!copiedStatus[msg.id]) e.currentTarget.style.opacity = '0.6';}}>
                                {copiedStatus[msg.id] ? <FiCheck size={14} style={{color: currentColors.success}} /> : <FiCopy size={14} />}
                            </button>
                        )}
                        {msg.sender === 'bot' && msg.sources && msg.sources.length > 0 && (
                            <div style={{ marginTop: '12px', paddingTop:'10px', borderTop: `1px solid ${isUser ? currentColors.userBubbleText+'33' : currentColors.border+'99'}` }}>
                                <button onClick={() => toggleSourceExpansion(msg.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 0 8px 0', fontSize: '0.8rem', fontWeight: '600', color: isUser ? currentColors.userBubbleText : currentColors.primary, display: 'flex', alignItems: 'center' }} >
                                    {expandedSourcesMap[msg.id] ? <FiChevronUp style={{marginRight:'5px'}} size={16}/> : <FiChevronDown style={{marginRight:'5px'}} size={16}/>}
                                    {expandedSourcesMap[msg.id] ? 'Ascunde Surse' : `Vezi Surse (${msg.sources.length})`}
                                </button>
                                {expandedSourcesMap[msg.id] && (
                                    <div style={{ marginTop: '5px', maxHeight:'200px', overflowY:'auto', padding: '8px', backgroundColor: isUser ? 'rgba(255,255,255,0.05)' : currentColors.sourceSnippetBg, borderRadius: '8px', border: `1px solid ${isUser ? 'rgba(255,255,255,0.1)' : currentColors.sourceSnippetBorder}` }}>
                                        {msg.sources.map((source, index) => (
                                            <div key={index} style={{ fontSize: '0.8rem', marginBottom: '8px', padding: '10px 12px', borderRadius: '8px', backgroundColor: currentColors.cardBackground, border: `1px solid ${currentColors.border}`, boxShadow: '0 1px 3px rgba(0,0,0,0.04)', transition: 'transform 0.2s ease, box-shadow 0.2s ease' }} onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.08)';}} onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.04)';}}>
                                                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom: '6px' }}>
                                                    <button onClick={() => handleSourceClick(source)} title={`Vezi detalii pentru ${source.metadata?.source}`} style={{ background: 'none', border: 'none', padding: 0, fontWeight: 600, color: currentColors.primary, cursor: 'pointer', fontSize: '0.85rem', textAlign: 'left', display: 'flex', alignItems: 'center', gap: '5px' }} onMouseEnter={(e) => e.currentTarget.style.textDecoration = 'underline'} onMouseLeave={(e) => e.currentTarget.style.textDecoration = 'none'} >
                                                      <FiFileText size={14} />
                                                      {index+1}. {source.metadata?.source || 'Necunoscută'}
                                                    </button>
                                                    {source.score !== undefined && source.score !== null && <span style={getScoreBadgeStyle(source.score)}> {source.score.toFixed(3)} </span> }
                                                </div>
                                                <p style={{ margin: '0', whiteSpace: 'normal', opacity: 0.85, color: currentColors.textMuted, fontSize:'0.8rem', lineHeight: '1.5' }}>
                                                    <em>"{source.page_content.substring(0, 200)}..."</em>
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div style={{ display: 'flex', padding: '0 2px', marginTop: '3px' }}>
                        {msg.timestamp && ( <p style={{fontSize: '0.65rem', color: currentColors.textMuted, margin: 0, marginRight: (msg.sender === 'bot' && msg.processingTime) ? '6px' : '0' }}> {formattedTimestamp} </p> )}
                        {msg.sender === 'bot' && msg.processingTime && ( <p style={{ fontSize: '0.65rem', color: currentColors.textMuted, margin: 0 }}> (Timp: {msg.processingTime.toFixed(1)}s) </p> )}
                    </div>
                </div>
                {userAvatar}
            </div>
          );
        })}

        {isLoading && (
          <div className={styles.messageAppear} style={{ display:'flex', justifyContent:'flex-start', paddingLeft:0}}>
            <div style={{ display:'flex', alignItems:'flex-start', gap:'10px' }}>
                <AvatarComponent sender="bot" /> 
                <div style={{ alignSelf: 'flex-start' }}>
                    <div style={{ padding: '10px 15px', borderRadius: '20px 20px 20px 5px', backgroundColor: currentColors.botBubbleBg, color: currentColors.botBubbleText, boxShadow: '0 2px 5px rgba(0,0,0,0.08)', display:'inline-block' }}>
                    <p style={{ margin: 0, fontStyle: 'italic', display: 'flex', alignItems: 'center' }}>
                        <FiLoader size={12} style={{ display: 'inline-block', animation: 'spin 1s linear infinite', marginRight: '6px' }}/>
                        <span>Bot-ul scrie...</span>
                    </p>
                    <style jsx global>{` @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } `}</style>
                    </div>
                </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} id="chat-form" style={{ display: 'flex', padding: '10px 12px', borderTop: `1px solid ${currentColors.border}`, backgroundColor: currentColors.cardBackground, flexShrink: 0, gap: '8px', alignItems: 'center' }}>
        <input type="text" value={currentQuestion} onChange={(e) => setCurrentQuestion(e.target.value)} placeholder="Scrie mesajul tău..." disabled={isLoading} style={{ flexGrow: 1, padding: '8px 12px', border: `1.5px solid ${currentColors.border}`, borderRadius: '16px', fontSize: '0.9rem', color: currentColors.textBody, backgroundColor: currentColors.inputBackground, outline: 'none', minHeight: '38px', boxSizing: 'border-box' }}
          onKeyPress={(event) => { if (event.key === 'Enter' && !event.shiftKey && !isLoading && currentQuestion.trim()) { event.preventDefault(); handleSubmit(event as any); } }} />
        <button type="submit" disabled={isLoading || !currentQuestion.trim()} style={{ padding: '0 16px', backgroundColor: (isLoading || !currentQuestion.trim()) ? currentColors.textMuted : currentColors.primary, color: 'white', border: 'none', borderRadius: '16px', fontSize: '0.9rem', fontWeight: 600, cursor: (isLoading || !currentQuestion.trim()) ? 'not-allowed' : 'pointer', transition: 'background-color 0.2s ease', display:'flex', alignItems:'center', justifyContent: 'center', gap:'6px', minHeight: '38px', boxSizing: 'border-box' }} >
          <FiSend size={15} /> <span>Trimite</span>
        </button>
      </form>
    </div>
  );
}