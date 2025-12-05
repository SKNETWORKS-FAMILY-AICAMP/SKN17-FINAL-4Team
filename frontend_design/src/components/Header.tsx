import React from 'react';
import { Lamp } from 'lucide-react';
import type { Page } from '../types/navigation';

interface HeaderProps {
  onNavigate: (page: Page) => void;
  isLoggedIn: boolean;
  onLogout?: () => void;
  currentPage?: Page;
}

export function Header({ onNavigate, isLoggedIn, onLogout, currentPage }: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/85 backdrop-blur-md border-b border-blue-100 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 py-[14px] flex items-center justify-between">

        {/* Logo */}
        <button
          onClick={() => onNavigate('chat')}
          className="flex items-center gap-2.5 hover:opacity-80 transition-opacity whitespace-nowrap"
        >
          <div className="w-9 h-9 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center shadow-md">
            <Lamp size={18} className="text-white" />
          </div>

          {/* 텍스트 절대 고정 */}
          <span className="text-[20px] font-medium leading-none bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent select-none">
            MOOD ON
          </span>
        </button>

        {/* Navigation */}
        <nav className="flex items-center gap-[10px] whitespace-nowrap">

          {isLoggedIn ? (
            <>
              <button
                onClick={() => onNavigate('mypage')}
                className="px-5 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 leading-none transition-colors"
              >
                마이페이지
              </button>

              <button
                onClick={() => onNavigate('reference')}
                className="px-5 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 leading-none transition-colors"
              >
                레퍼런스 보드
              </button>

              <button
                onClick={() => onNavigate('preference')}
                className="px-5 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 leading-none transition-colors"
              >
                취향분석
              </button>

              <button
                onClick={onLogout}
                className="px-5 py-2 text-[15px] bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-md leading-none"
              >
                로그아웃
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => onNavigate('signup')}
                className="px-5 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 leading-none transition-colors"
              >
                회원가입
              </button>

              <button
                onClick={() => onNavigate('login')}
                className="px-5 py-2 text-[15px] bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-md leading-none"
              >
                로그인
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}