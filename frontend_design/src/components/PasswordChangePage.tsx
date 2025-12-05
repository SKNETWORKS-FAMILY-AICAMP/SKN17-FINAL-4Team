import React, { useState } from 'react';
import { Lock, Eye, EyeOff, Lamp, AlertCircle, CheckCircle } from 'lucide-react';
import type { Page } from '../types/navigation';
import { api, initCsrfToken } from '../lib/api';

interface PasswordChangePageProps {
  onNavigate: (page: Page) => void;
}

export function PasswordChangePage({ onNavigate }: PasswordChangePageProps) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validatePassword = (pwd: string) => {
    // 6~16자
    if (pwd.length < 6 || pwd.length > 16) {
      return '비밀번호는 6~16자로 입력해주세요.';
    }

    // 영문, 숫자, 특수문자 중 2종류 이상
    const hasLetter = /[a-zA-Z]/.test(pwd);
    const hasNumber = /[0-9]/.test(pwd);
    const hasSpecial = /[!?~@#$%&^]/.test(pwd);
    const typesCount = [hasLetter, hasNumber, hasSpecial].filter(Boolean).length;

    if (typesCount < 2) {
      return '영문, 숫자, 특수문자(!?~@#$%&^) 중 2종류 이상을 혼용해주세요.';
    }

    // 동일한 숫자 3회 이상 반복 (111, 0000)
    if (/(\d)\1{2,}/.test(pwd)) {
      return '연속되거나 동일한 숫자는 사용할 수 없습니다.';
    }

    // 연속된 숫자 (012, 123, 234 등)
    if (/012|123|234|345|456|567|678|789|987|876|765|654|543|432|321|210/.test(pwd)) {
      return '연속되거나 동일한 숫자는 사용할 수 없습니다.';
    }

    // 동일한 문자 3회 이상 반복 (aaa, AAA)
    if (/([a-zA-Z])\1{2,}/.test(pwd)) {
      return '연속되거나 동일한 문자는 사용할 수 없습니다.';
    }

    // 연속된 문자 (abc, qwer 등)
    if (/abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|qwer|asdf|zxcv/.test(pwd.toLowerCase())) {
      return '연속되거나 동일한 문자는 사용할 수 없습니다.';
    }

    return '';
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
      if (typeof data?.old_password === 'string') {
        return data.old_password;
      }
      if (Array.isArray(data?.old_password)) {
        return data.old_password[0];
      }
      if (Array.isArray(data?.password)) {
        return data.password[0];
      }
    }
    if (err instanceof Error) {
      return err.message;
    }
    return '비밀번호를 변경하지 못했습니다. 잠시 후 다시 시도해주세요.';
  };

  const handleSubmit = async () => {
    setError('');

    if (!currentPassword) {
      setError('기존 비밀번호를 입력해주세요.');
      return;
    }

    if (!newPassword) {
      setError('새 비밀번호를 입력해주세요.');
      return;
    }

    const passwordError = validatePassword(newPassword);
    if (passwordError) {
      setError(passwordError);
      return;
    }

    if (!confirmPassword) {
      setError('새 비밀번호 확인을 입력해주세요.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('새 비밀번호가 일치하지 않습니다.');
      return;
    }

    if (currentPassword === newPassword) {
      setError('기존 비밀번호와 동일한 비밀번호는 사용할 수 없습니다.');
      return;
    }

    try {
      setIsSubmitting(true);
      await initCsrfToken();
      await api.post('/accounts/password/change/', {
        old_password: currentPassword,
        password: newPassword,
        password2: confirmPassword,
      });
      setShowSuccessPopup(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSuccessConfirm = () => {
    setShowSuccessPopup(false);
    onNavigate('mypage');
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
            onClick={() => onNavigate('mypage')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            마이페이지
          </button>
        </div>
      </div>

      <main className="h-[calc(100vh-80px)] flex items-center justify-center pt-30 py-4">
        <div className="max-w-md mx-auto px-6 w-full">
          <div className="bg-white rounded-3xl shadow-xl p-6">
            {/* Lock Icon */}
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Lock size={30} className="text-white" />
              </div>
            </div>

            <h1 className="text-xl mb-1.5 text-center bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              비밀번호 수정
            </h1>
            <p className="text-gray-500 text-center mb-4 text-sm">안전한 비밀번호를 설정해주세요</p>

            <div className="space-y-3 mb-4">
              {/* 기존 비밀번호 */}
              <div>
                <label className="block mb-1.5 text-gray-700 text-sm">기존 비밀번호</label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type={showCurrentPassword ? "text" : "password"}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="기존 비밀번호를 입력해주세요"
                    className="w-full pl-11 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showCurrentPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              {/* 새 비밀번호 */}
              <div>
                <label className="block mb-1.5 text-gray-700 text-sm">비밀번호</label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type={showNewPassword ? "text" : "password"}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="영문, 숫자, 특수문자 2종류 이상 혼용 (6~16자)"
                    className="w-full pl-11 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showNewPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              {/* 새 비밀번호 확인 */}
              <div>
                <label className="block mb-1.5 text-gray-700 text-sm">비밀번호 확인</label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="비밀번호를 다시 입력해주세요"
                    className="w-full pl-11 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>
            </div>

            {/* 비밀번호 조건 */}
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-2xl mb-3">
              <p className="text-xs text-gray-700 mb-1.5">비밀번호 조건:</p>
              <ul className="text-xs text-gray-600 space-y-0.5">
                <li>• 6~16자</li>
                <li>• 영문, 숫자, 특수문자(!?~@#$%&^) 중 2종류 이상</li>
                <li>• 연속되거나 동일한 문자/숫자 불가</li>
              </ul>
            </div>

            {error && (
              <div className="flex items-center gap-2 p-2.5 bg-red-50 border border-red-200 rounded-2xl mb-3">
                <AlertCircle size={16} className="text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-700 leading-relaxed">{error}</p>
              </div>
            )}

            {/* 버튼 */}
            <div className="flex gap-3">
              <button
                onClick={() => onNavigate('mypage')}
                className="flex-1 py-2.5 bg-gray-100 text-gray-700 rounded-2xl hover:bg-gray-200 transition-all text-sm"
              >
                취소
              </button>
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="flex-1 py-2.5 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg text-sm disabled:opacity-50"
              >
                {isSubmitting ? '변경 중...' : '변경하기'}
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Success Popup */}
      {showSuccessPopup && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl shadow-2xl p-6 max-w-sm w-full">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle size={32} className="text-green-500" />
              </div>
            </div>
            <h3 className="text-xl mb-2 text-center text-gray-800">비밀번호 변경 완료</h3>
            <p className="text-sm text-gray-600 text-center mb-6">
              비밀번호가 성공적으로 변경되었습니다.
            </p>
            <button
              onClick={handleSuccessConfirm}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
            >
              확인
            </button>
          </div>
        </div>
      )}
    </div>
  );
}