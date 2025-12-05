import React, { useState } from 'react';
import { Mail, Lock, AlertCircle, Eye, EyeOff, Lamp } from 'lucide-react';
import type { Page } from '../types/navigation';

interface LoginPageProps {
  onNavigate: (page: Page) => void;
  onLogin: (payload: { email: string; password: string }) => Promise<void>;
  isAuthenticating?: boolean;
}

export function LoginPage({ onNavigate, onLogin, isAuthenticating }: LoginPageProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateEmail = (email: string) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const extractErrorMessage = (err: unknown) => {
    if (
      err &&
      typeof err === 'object' &&
      'response' in err &&
      err.response &&
      typeof err.response === 'object' &&
      'data' in err.response
    ) {
      const data = (err.response as any).data;
      if (typeof data?.detail === 'string') {
        return data.detail;
      }
    }
    if (err instanceof Error) {
      return err.message;
    }
    return '로그인 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
  };

  const handleLogin = async () => {
    setError('');

    if (!validateEmail(email)) {
      setError('올바른 이메일 형식이 아닙니다.');
      return;
    }

    if (!email || !password) {
      setError('이메일과 비밀번호를 입력해주세요.');
      return;
    }

    try {
      setIsSubmitting(true);
      await onLogin({ email, password });
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLogin();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-yellow-50 overflow-y-scroll">
      {/* Custom Header */}
      <div className="border-b border-blue-100 px-5 py-3.5 flex items-center justify-between bg-white/80 backdrop-blur-sm shadow-sm fixed top-0 left-0 right-0 z-50">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('chat')}
            className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
          >
            <div className="w-9 h-9 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center shadow-md">
              <Lamp size={18} className="text-white" />
            </div>
            <span className="text-[20px] font-medium leading-none bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent select-none">
              MOOD ON
            </span>
          </button>
        </div>

        {/* Navigation Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('signup')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            회원가입
          </button>
          <button
            onClick={() => onNavigate('login')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            로그인
          </button>
        </div>
      </div>

      <main className="h-[calc(100vh-80px)] flex items-center justify-center pt-30 py6">
        <div className="max-w-md mx-auto px-6 w-full">
          <div className="bg-white rounded-3xl shadow-xl p-6">
            {/* Logo/Icon */}
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-white/50 via-blue-200/40 to-blue-400/30 backdrop-blur-lg shadow-2xl border border-white/40" 
                   style={{ boxShadow: 'inset -8px -8px 16px rgba(255, 255, 255, 0.6), inset 8px 8px 16px rgba(59, 130, 246, 0.15), 0 20px 40px rgba(59, 130, 246, 0.3)' }}>
              </div>
            </div>

            <h1 className="text-2xl mb-1 text-center bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              로그인
            </h1>
            <p className="text-gray-500 text-center mb-5 text-sm">MOOD ON에 오신 것을 환영합니다</p>

            <div className="space-y-3 mb-4">
              <div>
                <label className="block mb-1.5 text-gray-700 text-sm">이메일</label>
                <div className="relative">
                  <Mail size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="example@email.com"
                    disabled={isSubmitting || isAuthenticating}
                    className="w-full pl-11 pr-4 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 disabled:bg-gray-100 text-sm"
                  />
                </div>
              </div>

              <div>
                <label className="block mb-1.5 text-gray-700 text-sm">비밀번호</label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="비밀번호"
                    disabled={isSubmitting || isAuthenticating}
                    className="w-full pl-11 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 disabled:bg-gray-100 text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-2xl mb-4">
                <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-700 leading-relaxed">{error}</p>
              </div>
            )}

            <button
              onClick={handleLogin}
              disabled={isSubmitting || isAuthenticating}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed mb-3"
            >
              {isSubmitting ? '로그인 중...' : '로그인'}
            </button>

            <div className="flex items-center justify-between text-xs mb-4">
              <button
                onClick={() => onNavigate('password-reset')}
                className="text-blue-600 hover:underline"
              >
                비밀번호 찾기
              </button>
              <button
                onClick={() => onNavigate('signup')}
                className="text-blue-600 hover:underline"
              >
                회원가입
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}