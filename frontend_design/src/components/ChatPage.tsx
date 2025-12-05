import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Send,
  Image as ImageIcon,
  Plus,
  RefreshCcw,
  Trash2,
  Loader2,
  Camera,
  Heart,
  MessageCircle,
  Sparkles,
  FileText,
} from 'lucide-react';

import type { Page } from '../types/navigation';
import { api } from '../lib/api';

interface ChatPageProps {
  onNavigate: (page: Page) => void;
  isLoggedIn: boolean;
  chatSessions: ChatSessionSummary[];
  onRefreshSessions?: () => Promise<void> | void;
  onAddFavorite: (productId: number) => Promise<void> | void;
}

interface ChatSessionSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string | null;
  image_url?: string | null;
  recommended_products?: RecommendedProduct[];
  created_at: string;
  satisfaction?: number | null;
}

interface RecommendedProduct {
  product_id?: number;
  product_name?: string;
  name?: string;
  brand_name?: string;
  image_url?: string;
  link_url?: string;
  price?: number | string;
}

interface SessionStateData {
  category?: string | null;
  space?: string | null;
  price_min?: number | null;
  price_max?: number | null;
  target_moods?: string[] | null;
  current_moods?: string[] | null;
  style_keywords?: string[] | null;
  color_keywords?: string[] | null;
  material_keywords?: string[] | null;
  lighting_keywords?: string[] | null;
  vlm_description?: string | null;
  target_image_description?: string | null;
}

const SESSION_STATE_LIST_FIELDS: Array<
  'target_moods' | 'current_moods' | 'style_keywords' | 'color_keywords' | 'material_keywords' | 'lighting_keywords'
> = ['target_moods', 'current_moods', 'style_keywords', 'color_keywords', 'material_keywords', 'lighting_keywords'];
const SESSION_STATE_FIELDS: (keyof SessionStateData)[] = [
  'category',
  'space',
  'price_min',
  'price_max',
  'target_moods',
  'current_moods',
  'style_keywords',
  'color_keywords',
  'material_keywords',
  'lighting_keywords',
  'vlm_description',
  'target_image_description',
];

export function ChatPage({
  onNavigate,
  isLoggedIn,
  chatSessions,
  onRefreshSessions,
  onAddFavorite,
}: ChatPageProps) {
  const [activeSessionId, setActiveSessionId] = useState<number | null>(
    chatSessions.length ? chatSessions[0].id : null,
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionState, setSessionState] = useState<SessionStateData | null>(null);
  const [inputText, setInputText] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [error, setError] = useState('');
  const [stateMessage, setStateMessage] = useState('');
  const [stateMessageTone, setStateMessageTone] = useState<'success' | 'error'>('success');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const stateMessageTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionList = chatSessions;

  useEffect(() => {
    if (!isLoggedIn) {
      setActiveSessionId(null);
      setMessages([]);
      setSessionState(null);
      return;
    }
    if (chatSessions.length === 0) {
      setActiveSessionId(null);
      onRefreshSessions?.();
      return;
    }
    if (activeSessionId === null || !chatSessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(chatSessions[0].id);
    }
  }, [chatSessions, activeSessionId, isLoggedIn, onRefreshSessions]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    return () => {
      if (stateMessageTimer.current) {
        clearTimeout(stateMessageTimer.current);
      }
    };
  }, []);

  const showStateFeedback = useCallback(
    (text: string, tone: 'success' | 'error' = 'success') => {
      if (stateMessageTimer.current) {
        clearTimeout(stateMessageTimer.current);
      }
      setStateMessageTone(tone);
      setStateMessage(text);
      stateMessageTimer.current = setTimeout(() => {
        setStateMessage('');
      }, 3000);
    },
    [],
  );

  const fetchSessionDetail = useCallback(
    async (sessionId: number) => {
      if (!sessionId) return;
      setIsLoadingMessages(true);
      try {
        const res = await api.get(`/chat/sessions/${sessionId}/`);
        setMessages(res.data?.messages ?? []);
        setSessionState(res.data?.state ?? {});
      } catch (err) {
        console.error('세션 정보를 불러오지 못했습니다.', err);
      } finally {
        setIsLoadingMessages(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (activeSessionId && isLoggedIn) {
      fetchSessionDetail(activeSessionId);
    } else {
      setMessages([]);
      setSessionState(null);
    }
  }, [activeSessionId, fetchSessionDetail, isLoggedIn]);

  const handleCreateSession = useCallback(async () => {
    if (!isLoggedIn) {
      setError('로그인 후 이용 가능합니다.');
      return null;
    }
    try {
      const res = await api.post('/chat/sessions/');
      await onRefreshSessions?.();
      if (res.data?.id) {
        setActiveSessionId(res.data.id);
        return res.data.id as number;
      }
    } catch (err) {
      console.error('새 채팅 생성 실패', err);
    }
    return null;
  }, [isLoggedIn, onRefreshSessions]);

  const ensureSession = useCallback(async () => {
    if (activeSessionId) return activeSessionId;
    const newId = await handleCreateSession();
    if (newId) {
      setActiveSessionId(newId);
      return newId;
    }
    return null;
  }, [activeSessionId, handleCreateSession]);

  const handleDeleteSession = async (sessionId: number) => {
    if (!window.confirm('선택한 세션을 삭제하시겠습니까?')) return;
    try {
      await api.delete(`/chat/sessions/${sessionId}/`);
      await onRefreshSessions?.();
      if (sessionId === activeSessionId) {
        setActiveSessionId(null);
        setMessages([]);
        setSessionState(null);
      }
    } catch (err) {
      console.error('세션 삭제 실패', err);
    }
  };

  const handleResetSession = async (sessionId: number) => {
    if (!window.confirm('현재 세션을 초기화하시겠습니까?')) return;
    try {
      await api.post(`/chat/sessions/${sessionId}/reset/`);
      if (sessionId === activeSessionId) {
        await fetchSessionDetail(sessionId);
      }
      await onRefreshSessions?.();
    } catch (err) {
      console.error('세션 초기화 실패', err);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.match(/(jpeg|jpg|png)$/i)) {
      setError('JPG 또는 PNG 이미지만 업로드할 수 있습니다.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('이미지는 10MB 이하여야 합니다.');
        return;
    }
    setError('');
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const sendMessage = async (options?: { requestMore?: boolean; textOverride?: string }) => {
    if (!isLoggedIn) {
      setError('로그인 후 이용 가능합니다.');
      return;
    }
    const sessionId = activeSessionId ?? (await ensureSession());
    if (!sessionId) {
      setError('세션을 생성하지 못했습니다. 잠시 후 다시 시도해주세요.');
      return;
    }
    if (!inputText && !imageFile && !options?.textOverride) {
      setError('메시지를 입력하거나 이미지를 업로드해주세요.');
      return;
    }

    setError('');
    setIsSending(true);
    try {
      const formData = new FormData();
      if (options?.textOverride) {
        formData.append('text', options.textOverride);
      } else if (inputText) {
        formData.append('text', inputText);
      }
      if (imageFile) {
        formData.append('image', imageFile);
      }
      if (options?.requestMore) {
        formData.append('request_more', 'true');
      }
      await api.post(`/chat/sessions/${sessionId}/messages/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setInputText('');
      setImageFile(null);
      setImagePreview(null);
      await fetchSessionDetail(sessionId);
      await onRefreshSessions?.();
    } catch (err) {
      console.error('메시지를 전송하지 못했습니다.', err);
      setError('메시지를 보낼 수 없습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsSending(false);
    }
  };

  const handleRequestMore = () => {
    sendMessage({ requestMore: true, textOverride: '다른 상품을 추천해줘요' });
  };

  const handleSatisfaction = async (messageId: number, score: number) => {
    try {
      await api.post(`/chat/messages/${messageId}/satisfaction/`, { score });
      setMessages((prev) =>
        prev.map((msg) => (msg.id === messageId ? { ...msg, satisfaction: score } : msg)),
      );
    } catch (err) {
      console.error('만족도 저장 실패', err);
    }
  };

  const handleSessionStateChange = (field: keyof SessionStateData, value: string) => {
    setSessionState((prev) => ({
      ...(prev ?? {}),
      [field]: value.length ? value : null,
    }));
  };

  const handleSessionStateNumber = (field: 'price_min' | 'price_max', value: string) => {
    const parsed = value ? Number(value) : null;
    setSessionState((prev) => ({
      ...(prev ?? {}),
      [field]: parsed,
    }));
  };

  const handleSessionStateListChange = (
    field: (typeof SESSION_STATE_LIST_FIELDS)[number],
    value: string,
  ) => {
    const items = value
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean);
    setSessionState((prev) => ({
      ...(prev ?? {}),
      [field]: items,
    }));
  };

  const handleSaveState = async () => {
    if (!activeSessionId || !sessionState) return;
    try {
      const res = await api.patch(`/chat/sessions/${activeSessionId}/state/`, sessionState);
      setSessionState(res.data ?? sessionState);
      showStateFeedback('세션 상태가 저장되었습니다.', 'success');
    } catch (err) {
      console.error('세션 상태 저장 실패', err);
      showStateFeedback('상태를 저장하지 못했습니다.', 'error');
    }
  };

  const handleClearState = async () => {
    if (!activeSessionId) return;
    const resetPayload = SESSION_STATE_FIELDS.reduce((acc, key) => {
      if (SESSION_STATE_LIST_FIELDS.includes(key as (typeof SESSION_STATE_LIST_FIELDS)[number])) {
        acc[key] = [];
      } else {
        acc[key] = null;
      }
      return acc;
    }, {} as Partial<Record<keyof SessionStateData, any>>);
    try {
      const res = await api.patch(`/chat/sessions/${activeSessionId}/state/`, resetPayload);
      setSessionState(res.data ?? {});
      showStateFeedback('세션 상태를 초기화했습니다.', 'success');
    } catch (err) {
      console.error('세션 상태 초기화 실패', err);
      showStateFeedback('상태를 초기화하지 못했습니다.', 'error');
    }
  };

  const renderProducts = (products?: RecommendedProduct[]) => {
    if (!products || products.length === 0) return null;
  return (
      <div className="mt-4 space-y-3">
        {products.map((product, idx) => {
          const productId = product.product_id;
          const name = product.product_name || product.name || '추천 상품';
          const brand = product.brand_name || '';
          const priceValue =
            typeof product.price === 'number'
              ? product.price
              : product.price
              ? parseInt(String(product.price).replace(/[^\d]/g, ''), 10)
              : undefined;
          const displayPrice = priceValue ? `₩${priceValue.toLocaleString()}` : '가격 정보 없음';
          return (
            <div key={productId ?? idx} className="rounded-2xl border border-blue-100 p-3">
              <div className="flex gap-3">
                {product.image_url && (
                  <img
                    src={product.image_url}
                    alt={name}
                    className="h-16 w-16 rounded-xl object-cover"
                  />
                )}
                <div className="flex-1 text-sm">
                  <p className="font-semibold text-gray-800">{name}</p>
                  {brand && <p className="text-xs text-gray-500">{brand}</p>}
                  <p className="text-xs text-gray-500">{displayPrice}</p>
                  <div className="mt-2 flex gap-2">
                    {product.link_url && (
                      <a
                        href={product.link_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                        className="rounded-full border border-blue-300 px-3 py-1 text-xs text-blue-600 hover:bg-blue-50"
                                  >
                        상품 보기
                                  </a>
                    )}
                    {productId && (
                                  <button
                        onClick={() => onAddFavorite(productId)}
                        className="flex items-center gap-1 rounded-full bg-pink-500 px-3 py-1 text-xs text-white hover:bg-pink-600"
                                  >
                        <Heart size={14} /> 관심상품
                                  </button>
                        )}
                      </div>
                  </div>
              </div>
                    </div>
          );
        })}
                  </div>
    );
  };

  const renderMessageTime = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  const renderMessages = () => {
    if (isLoadingMessages) {
      return (
        <div className="flex flex-1 items-center justify-center text-gray-500">
          <Loader2 size={28} className="animate-spin" />
                  </div>
      );
    }

    if (messages.length === 0) {
      return (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-gray-400">
          <MessageCircle size={32} />
          {isLoggedIn ? (
            <p>대화를 시작해 보세요.</p>
          ) : (
            <div className="text-center text-sm">
              <p className="mb-2">로그인 후 챗봇을 이용할 수 있습니다.</p>
                      <button
                        onClick={() => onNavigate('login')}
                className="rounded-full bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
                      >
                        로그인
                      </button>
                    </div>
                  )}
                </div>
      );
    }

    return (
      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
                    {messages.map((message) => (
          <div key={message.id} className="space-y-2">
            <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[72%] rounded-3xl px-4 py-3 text-sm ${
                  message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-gray-800 shadow'
                }`}
              >
                {message.text && <p className="whitespace-pre-line">{message.text}</p>}
                {message.image_url && (
                  <img
                    src={message.image_url}
                    alt="대화 이미지"
                    className="mt-3 max-h-60 rounded-2xl object-cover"
                  />
                )}
                <p
                  className={`mt-2 text-xs ${
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}
                >
                  {renderMessageTime(message.created_at)}
                            </p>
                          </div>
                        </div>

            {message.role === 'assistant' && message.recommended_products?.length ? (
              <div className="rounded-3xl border border-blue-100 bg-white p-4">
                <p className="text-sm font-semibold text-gray-800">추천 상품</p>
                {renderProducts(message.recommended_products)}
                <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                  <span>추천 결과는 도움이 되었나요?</span>
                  {[1, 2, 3, 4, 5].map((score) => (
                                    <button
                      key={score}
                      onClick={() => handleSatisfaction(message.id, score)}
                      className={`rounded-full border px-2 ${
                        message.satisfaction === score
                          ? 'border-blue-400 text-blue-600'
                          : 'border-gray-200 text-gray-500'
                      }`}
                    >
                      {score}
                                    </button>
                  ))}
                              <button
                    onClick={handleRequestMore}
                    className="ml-auto rounded-full border border-gray-200 px-3 py-1 text-xs text-gray-600 hover:bg-gray-50"
                  >
                    다른 상품 보기
                                </button>
                              </div>
                            </div>
            ) : null}
                      </div>
                    ))}
        <div ref={messagesEndRef} />
                  </div>
    );
  };

  return (
    <div className="flex min-h-screen bg-white">
      <aside className="w-72 border-r border-gray-100 bg-gray-50">
        <div className="sticky top-0 border-b border-gray-100 bg-gray-50 px-4 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-800">채팅 히스토리</h2>
                      <button
              onClick={handleCreateSession}
              className="flex items-center gap-1 rounded-full bg-blue-500 px-3 py-1.5 text-xs text-white hover:bg-blue-600 disabled:opacity-40"
              disabled={!isLoggedIn}
                      >
              <Plus size={14} /> 새 채팅
                      </button>
                    </div>
        </div>
        <div className="space-y-2 overflow-y-auto px-4 py-4">
          {sessionList.length === 0 && (
            <div className="rounded-2xl border border-dashed border-gray-300 p-4 text-center text-sm text-gray-500">
              아직 대화가 없습니다.
                  </div>
                )}
          {sessionList.map((session) => (
            <div
              key={session.id}
              className={`rounded-2xl border px-3 py-2 shadow-sm transition ${
                activeSessionId === session.id
                  ? 'border-blue-400 bg-white'
                  : 'border-transparent bg-white/80 hover:border-gray-200'
              }`}
            >
              <button onClick={() => setActiveSessionId(session.id)} className="text-left">
                <p className="text-sm font-semibold text-gray-800">{session.title}</p>
                <p className="text-[11px] text-gray-500">
                  {new Date(session.created_at).toLocaleString('ko-KR')}
                </p>
                  </button>
              <div className="mt-2 flex gap-1 text-[11px] text-gray-500">
                  <button
                  onClick={() => handleResetSession(session.id)}
                  className="rounded-full border border-gray-200 px-2 py-0.5 hover:text-blue-600"
                >
                  <RefreshCcw size={11} /> 초기화
                  </button>
                        <button
                  onClick={() => handleDeleteSession(session.id)}
                  className="rounded-full border border-gray-200 px-2 py-0.5 hover:text-red-600"
                        >
                  <Trash2 size={11} /> 삭제
                        </button>
                    </div>
                  </div>
          ))}
              </div>
      </aside>

      <div className="flex flex-1 flex-col bg-gradient-to-br from-white to-blue-50/40">
        <header className="sticky top-0 border-b border-gray-100 bg-white/90 px-6 py-4 backdrop-blur">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">MOOD ON 챗봇</h1>
              <p className="text-sm text-gray-500">무드 기반 인테리어 소품 추천</p>
              </div>
            <div className="flex gap-2">
              <button
                onClick={() => onNavigate('reference')}
                className="rounded-full border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50"
              >
                레퍼런스 보드
              </button>
              <button
                onClick={() => onNavigate('mypage')}
                className="rounded-full bg-blue-500 px-4 py-2 text-sm text-white hover:bg-blue-600"
              >
                마이페이지
              </button>
            </div>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <section className="flex-1 flex flex-col">{renderMessages()}</section>

          <aside className="w-80 border-l border-gray-100 bg-white/80 px-5 py-6">
            <div className="rounded-3xl border border-gray-100 bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-gray-100 pb-3">
                <div className="flex items-center gap-2">
                  <FileText size={18} className="text-blue-500" />
                  <span className="text-sm font-semibold text-gray-800">세션 상태</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleClearState}
                    className="flex items-center gap-1 rounded-full border border-gray-200 px-2 py-1 text-xs text-gray-500 hover:bg-gray-50 disabled:opacity-40"
                    disabled={!sessionState}
                  >
                    <RefreshCcw size={12} />
                    초기화
                  </button>
                  <button
                    onClick={handleSaveState}
                    className="flex items-center gap-1 rounded-full bg-blue-50 px-3 py-1 text-xs text-blue-600 hover:bg-blue-100 disabled:opacity-40"
                    disabled={!sessionState}
                  >
                    <Sparkles size={12} />
                    저장
                  </button>
                </div>
              </div>

              {sessionState ? (
                <div className="space-y-3 pt-3 text-sm">
                  <label className="block text-xs text-gray-500">
                    카테고리
                    <input
                      value={sessionState.category ?? ''}
                      onChange={(e) => handleSessionStateChange('category', e.target.value)}
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <label className="block text-xs text-gray-500">
                    공간
                    <input
                      value={sessionState.space ?? ''}
                      onChange={(e) => handleSessionStateChange('space', e.target.value)}
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="text-xs text-gray-500">
                      최소 예산
                      <input
                        type="number"
                        value={sessionState.price_min ?? ''}
                        onChange={(e) => handleSessionStateNumber('price_min', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                    <label className="text-xs text-gray-500">
                      최대 예산
                      <input
                        type="number"
                        value={sessionState.price_max ?? ''}
                        onChange={(e) => handleSessionStateNumber('price_max', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                  </div>
                  <label className="block text-xs text-gray-500">
                    타겟 무드 (쉼표로 구분)
                    <textarea
                      value={(sessionState.target_moods ?? []).join(', ')}
                      onChange={(e) => handleSessionStateListChange('target_moods', e.target.value)}
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <label className="block text-xs text-gray-500">
                    현재 무드 (쉼표로 구분)
                    <textarea
                      value={(sessionState.current_moods ?? []).join(', ')}
                      onChange={(e) => handleSessionStateListChange('current_moods', e.target.value)}
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="text-xs text-gray-500">
                      스타일 키워드
                      <textarea
                        value={(sessionState.style_keywords ?? []).join(', ')}
                        onChange={(e) => handleSessionStateListChange('style_keywords', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                    <label className="text-xs text-gray-500">
                      색상 키워드
                      <textarea
                        value={(sessionState.color_keywords ?? []).join(', ')}
                        onChange={(e) => handleSessionStateListChange('color_keywords', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                    <label className="text-xs text-gray-500">
                      재질 키워드
                      <textarea
                        value={(sessionState.material_keywords ?? []).join(', ')}
                        onChange={(e) => handleSessionStateListChange('material_keywords', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                    <label className="text-xs text-gray-500">
                      조명 키워드
                      <textarea
                        value={(sessionState.lighting_keywords ?? []).join(', ')}
                        onChange={(e) => handleSessionStateListChange('lighting_keywords', e.target.value)}
                        className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                      />
                    </label>
                  </div>
                  <label className="block text-xs text-gray-500">
                    VLM 설명
                    <textarea
                      rows={3}
                      value={sessionState.vlm_description ?? ''}
                      onChange={(e) => handleSessionStateChange('vlm_description', e.target.value)}
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <label className="block text-xs text-gray-500">
                    대표 이미지 설명
                    <textarea
                      rows={3}
                      value={sessionState.target_image_description ?? ''}
                      onChange={(e) =>
                        handleSessionStateChange('target_image_description', e.target.value)
                      }
                      className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 text-xs focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  {stateMessage && (
                    <p
                      className={`text-xs ${
                        stateMessageTone === 'success' ? 'text-green-600' : 'text-red-500'
                      }`}
                      aria-live="polite"
                    >
                      {stateMessage}
                    </p>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-12 text-sm text-gray-400">
                  <Sparkles size={20} />
                  세션을 선택하면 상태를 편집할 수 있어요.
                </div>
              )}
            </div>
          </aside>
            </div>
          </div>
    </div>
  );
}

