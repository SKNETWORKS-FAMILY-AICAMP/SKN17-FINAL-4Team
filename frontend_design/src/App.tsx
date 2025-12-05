import React, { useState, useEffect, useCallback } from 'react';
import { LoginPage } from './components/LoginPage';
import { SignUpPage } from './components/SignUpPage';
import { MyPage } from './components/MyPage';
import { ChatPage } from './components/ChatPage';
import { ChatGuidelinesPage } from './components/ChatGuidelinesPage';
import { PreferencePage } from './components/PreferencePage';
import { ReferenceBoard } from './components/ReferenceBoard';
import { PasswordResetPage } from './components/PasswordResetPage';
import { PasswordChangePage } from './components/PasswordChangePage';
import type { Page } from './types/navigation';
import { useAuth } from './hooks/useAuth';
import { api } from './lib/api';

type ChatSessionSummary = {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
};

type FavoriteProductRecord = {
  id: number;
  product: {
    id: number;
    brand_name: string;
    product_name: string;
    image_url: string;
    link_url: string;
    price: number;
  };
  created_at: string;
};

type UserPreferencesState = {
  gender?: string | null;
  birthdate?: string | null;
  mbti?: string | null;
  styles?: string[];
  preferred_moods?: string[];
};

const PROTECTED_PAGES: ReadonlyArray<Page> = ['mypage', 'preference', 'reference'];
// TODO(auth-e2e): 로그인하지 않은 사용자가 PROTECTED_PAGES 로 직접 진입하려고 할 때 guard 가 동작하는지 Cypress/E2E 테스트 추가 예정.

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('chat');
  const [userPreferences, setUserPreferences] = useState<UserPreferencesState | null>(null);
  const [favoriteProducts, setFavoriteProducts] = useState<FavoriteProductRecord[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSessionSummary[]>([]);
  const [showLoginRequiredPopup, setShowLoginRequiredPopup] = useState(false);
  const { user, isAuthenticated, initializing, login, logout, deleteAccount, refreshSession } = useAuth();
  const userEmail = user?.email ?? '';

  const resetAppState = useCallback(() => {
    setUserPreferences(null);
    setFavoriteProducts([]);
    setChatSessions([]);
  }, []);

  const fetchProfile = useCallback(async () => {
    if (!isAuthenticated) {
      resetAppState();
      return;
    }
    try {
      const [profileRes, preferenceRes, favoritesRes, sessionsRes] = await Promise.all([
        api.get('/accounts/profile/'),
        api.get('/favorites/preferences/'),
        api.get('/favorites/'),
        api.get('/chat/sessions/'),
      ]);
      const moods = preferenceRes.data?.preferred_moods ?? [];
      setUserPreferences({
        gender: profileRes.data.gender,
        birthdate: profileRes.data.birth_date,
        mbti: profileRes.data.mbti,
        styles: moods,
        preferred_moods: moods,
      });
      setFavoriteProducts(favoritesRes.data);
      setChatSessions(sessionsRes.data);
    } catch (error) {
      console.error('초기 데이터를 불러오지 못했습니다.', error);
    }
  }, [isAuthenticated, resetAppState]);

  const refreshFavorites = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const res = await api.get('/favorites/');
      setFavoriteProducts(res.data);
    } catch (error) {
      console.error('관심 상품을 불러오지 못했습니다.', error);
    }
  }, [isAuthenticated]);

  const refreshChatSessions = useCallback(async () => {
    if (!isAuthenticated) {
      setChatSessions([]);
      return;
    }
    try {
      const res = await api.get('/chat/sessions/');
      setChatSessions(res.data);
    } catch (error) {
      console.error('채팅 세션을 불러오지 못했습니다.', error);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const handleLogin = async (credentials: { email: string; password: string }) => {
    await login(credentials);
    await fetchProfile();
    setCurrentPage('chat');
  };

  const handleLogout = useCallback(async () => {
    // TODO(auth-cleanup): QA 체크 시 로그아웃/회원탈퇴 후 로컬 상태 및 브라우저 저장소가 모두 비워지는지 수동 테스트할 것.
    await logout();
    setCurrentPage('chat');
    setUserPreferences(null);
    setFavoriteProducts([]);
    setChatSessions([]);
  }, [logout]);

  const handlePreferenceComplete = (preferences: UserPreferencesState) => {
    const moods = preferences.preferred_moods ?? preferences.styles ?? [];
    setUserPreferences(prev => ({
      ...prev,
      ...preferences,
      styles: moods,
      preferred_moods: moods,
    }));
    setCurrentPage('reference');
  };

  const handleAddFavorite = async (productId: number) => {
    if (!isAuthenticated) return;
    try {
      await api.post('/favorites/', { product_id: productId });
      await refreshFavorites();
    } catch (error) {
      console.error('관심 상품 등록에 실패했습니다.', error);
    }
  };

  const isProtectedPage = useCallback((page: Page) => PROTECTED_PAGES.includes(page), []);

  const handleNavigate = (page: Page) => {
    if (!isAuthenticated && isProtectedPage(page)) {
      setShowLoginRequiredPopup(true);
      return;
    }
    setCurrentPage(page);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'main':
        return (
          <ChatPage
            onNavigate={handleNavigate}
            isLoggedIn={isAuthenticated}
            chatSessions={chatSessions}
            onRefreshSessions={refreshChatSessions}
            onAddFavorite={handleAddFavorite}
          />
        );
      case 'login':
        return <LoginPage onNavigate={handleNavigate} onLogin={handleLogin} isAuthenticating={initializing} />;
      case 'signup':
        return <SignUpPage onNavigate={handleNavigate} onSignUp={() => handleNavigate('login')} />;
      case 'password-reset':
        return <PasswordResetPage onNavigate={handleNavigate} />;
      case 'password-change':
        return <PasswordChangePage onNavigate={handleNavigate} />;
      case 'preference':
        return (
          <PreferencePage
            onNavigate={handleNavigate}
            onComplete={handlePreferenceComplete}
            onLogout={handleLogout}
            initialPreferences={userPreferences}
          />
        );
      case 'reference':
        return (
          <ReferenceBoard
            onNavigate={handleNavigate}
            isLoggedIn={isAuthenticated}
            userPreferences={userPreferences}
            onLogout={handleLogout}
            userEmail={userEmail}
          />
        );
      case 'chat':
        return (
          <ChatPage
            onNavigate={handleNavigate}
            isLoggedIn={isAuthenticated}
            chatSessions={chatSessions}
            onRefreshSessions={refreshChatSessions}
            onAddFavorite={handleAddFavorite}
          />
        );
      case 'chat-guidelines':
        return <ChatGuidelinesPage onNavigate={handleNavigate} isLoggedIn={isAuthenticated} onLogout={handleLogout} />;
      case 'mypage':
        return (
          <MyPage
            onNavigate={handleNavigate}
            onLogout={handleLogout}
            userEmail={userEmail}
            onDeleteAccount={deleteAccount}
            initialPreferences={userPreferences}
            initialFavorites={favoriteProducts}
            onRefreshUserData={fetchProfile}
          />
        );
      default:
        return (
          <ChatPage
            onNavigate={handleNavigate}
            isLoggedIn={isAuthenticated}
            chatSessions={chatSessions}
            onRefreshSessions={refreshChatSessions}
            onAddFavorite={handleAddFavorite}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {renderPage()}
      
      {/* Login Required Popup */}
      {showLoginRequiredPopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
            <div className="text-center mb-6">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
                <span className="text-4xl">🔒</span>
              </div>
              <h2 className="text-2xl mb-3 text-gray-800">
                로그인이 필요합니다
              </h2>
              <p className="text-gray-600">
                해당 서비스를 이용하시려면<br/>
                먼저 로그인해주세요.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setShowLoginRequiredPopup(false)}
                className="flex-1 py-4 border-2 border-gray-300 rounded-2xl hover:bg-gray-50 transition-all text-gray-700"
              >
                취소
              </button>
              <button
                onClick={() => {
                  setShowLoginRequiredPopup(false);
                  setCurrentPage('login');
                }}
                className="flex-1 py-4 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
              >
                로그인하기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}