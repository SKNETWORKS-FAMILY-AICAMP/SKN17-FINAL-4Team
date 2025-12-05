import React, { useState, useEffect } from 'react';
import { Check, Mail, Lock, AlertCircle, Clock, Eye, EyeOff, Lamp } from 'lucide-react';
import type { Page } from '../types/navigation';
import { api, initCsrfToken } from '../lib/api';

interface PasswordResetPageProps {
  onNavigate: (page: Page) => void;
}

export function PasswordResetPage({ onNavigate }: PasswordResetPageProps) {
  const [step, setStep] = useState<1 | 2>(1);
  const [email, setEmail] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [timeLeft, setTimeLeft] = useState(0);
  const [sendCount, setSendCount] = useState(0);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [error, setError] = useState('');
  const [showCodeSentPopup, setShowCodeSentPopup] = useState(false);
  const [showResetCompletePopup, setShowResetCompletePopup] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSendingCode, setIsSendingCode] = useState(false);
  const [isVerifyingCode, setIsVerifyingCode] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [codeRequested, setCodeRequested] = useState(false);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [timeLeft]);

  useEffect(() => {
    if (resendCooldown > 0) {
      const timer = setTimeout(() => setResendCooldown(resendCooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [resendCooldown]);

  const validateEmail = (email: string) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

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
      if (Array.isArray(data?.email)) {
        return data.email[0];
      }
      if (Array.isArray(data?.password)) {
        return data.password[0];
      }
    }
    if (err instanceof Error) {
      return err.message;
    }
    return '요청을 처리하지 못했습니다. 잠시 후 다시 시도해주세요.';
  };

  const handleSendCode = async () => {
    setError('');

    if (!validateEmail(email)) {
      setError('올바른 이메일 형식이 아닙니다.');
      return;
    }

    if (sendCount > 0 && resendCooldown > 0) {
      setError(`잠시 후 다시 시도해주세요. (${resendCooldown}초)`);
      return;
    }

    try {
      setIsSendingCode(true);
      await initCsrfToken();
      await api.post('/accounts/password/reset/email/', { email });
      setTimeLeft(180);
      setSendCount((prev) => prev + 1);
      setResendCooldown(10);
      setCodeRequested(true);
      setShowCodeSentPopup(true);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSendingCode(false);
    }
  };

  const handleVerifyCode = async () => {
    setError('');

    if (!codeRequested) {
      setError('먼저 인증번호를 요청해주세요.');
      return;
    }

    if (timeLeft === 0) {
      setError('인증 시간이 만료되었습니다. 인증번호를 재발송해주세요.');
      return;
    }

    try {
      setIsVerifyingCode(true);
      await initCsrfToken();
      await api.post('/accounts/password/reset/verify/', {
        email,
        code: verificationCode,
      });
      setStep(2);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsVerifyingCode(false);
    }
  };

  const handleCompleteReset = async () => {
    setError('');

    const pwdError = validatePassword(password);
    if (pwdError) {
      setError(pwdError);
      return;
    }

    if (password !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다.');
      return;
    }

    try {
      setIsResetting(true);
      await initCsrfToken();
      await api.post('/accounts/password/reset/complete/', {
        email,
        password,
        password2: confirmPassword,
      });
      setShowResetCompletePopup(true);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsResetting(false);
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

      <main className="h-[calc(100vh-80px)] flex justify-center items-start pt-30 pb-6">
        <div className="max-w-md mx-auto px-6 w-full">
          {/* Progress - 3단계로 표시 */}
          <div className="flex items-center justify-center mb-6">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center transition-all text-sm ${
                    (step === 1 && s === 1) || (step === 2 && s <= 2) || s === 3
                      ? step > s || (step === 2 && s === 1)
                        ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white shadow-lg'
                        : step >= s
                        ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white shadow-lg'
                        : 'bg-gray-200 text-gray-400'
                      : 'bg-gray-200 text-gray-400'
                  }`}
                >
                  {(step === 2 && s === 1) ? <Check size={16} /> : s}
                </div>
                {s < 3 && (
                  <div
                    className={`w-12 h-1 mx-1.5 transition-all ${
                      (step === 2 && s === 1)
                        ? 'bg-gradient-to-r from-blue-500 to-blue-400'
                        : 'bg-gray-200'
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Step 1: 이메일 인증 */}
          {step === 1 && (
            <div className="bg-white rounded-3xl shadow-xl p-6">
              <div className="flex items-center justify-center mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center">
                  <Mail size={24} className="text-white" />
                </div>
              </div>

              <h1 className="text-2xl mb-1 text-center bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                이메일 인증
              </h1>
              <p className="text-gray-500 text-center mb-5 text-sm">
                이메일로 인증번호를 받아주세요
              </p>

              <div className="space-y-3 mb-4">
                <div>
                  <label className="block mb-1.5 text-gray-700 text-sm">이메일</label>
                  <div className="flex gap-2">
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="example@email.com"
                      className="flex-1 px-3.5 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
                    />
                    <button
                      onClick={handleSendCode}
                      disabled={isSendingCode}
                      className="px-5 py-2.5 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all whitespace-nowrap disabled:opacity-50 text-sm"
                    >
                      {isSendingCode ? '발송 중...' : sendCount === 0 ? '발송' : '재발송'}
                    </button>
                  </div>
                  {sendCount > 0 && (
                    <p className="text-xs text-gray-500 mt-1.5">발송 횟수: {sendCount}/5</p>
                  )}
                </div>

                {codeRequested && (
                  <div>
                    <label className="block mb-1.5 text-gray-700 text-sm">인증번호</label>
                    <input
                      type="text"
                      value={verificationCode}
                      onChange={(e) => setVerificationCode(e.target.value)}
                      placeholder="인증번호 8자리"
                      maxLength={8}
                      className="w-full px-3.5 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
                    />
                    {timeLeft > 0 && (
                      <div className="flex items-center gap-2 mt-1.5 text-xs text-blue-600">
                        <Clock size={14} />
                        <span>
                          {Math.floor(timeLeft / 60)}:
                          {(timeLeft % 60).toString().padStart(2, '0')}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-2xl mb-4">
                  <AlertCircle
                    size={18}
                    className="text-red-500 flex-shrink-0"
                  />
                  <p className="text-xs text-red-700 leading-relaxed">{error}</p>
                </div>
              )}

              <button
                  onClick={handleVerifyCode}
                  disabled={!verificationCode || !codeRequested || isVerifyingCode}
                  className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                  {isVerifyingCode ? '확인 중...' : '인증하기'}
              </button>
            </div>
          )}

          {/* Step 2: 비밀번호 재설정 */}
          {step === 2 && (
            <div className="bg-white rounded-3xl shadow-xl p-6">
              <div className="flex items-center justify-center mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center">
                  <Lock size={24} className="text-white" />
                </div>
              </div>

              <h1 className="text-2xl mb-1 text-center bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                비밀번호 설정
              </h1>
              <p className="text-gray-500 text-center mb-5 text-sm">
                안전한 비밀번호를 설정해주세요
              </p>

              <div className="space-y-3 mb-4">
                <div>
                  <label className="block mb-1.5 text-gray-700 text-sm">비밀번호</label>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="영문, 숫자, 특수문자 2종류 이상 혼용 (6~16자)"
                      className="w-full px-3.5 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
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

                <div>
                  <label className="block mb-1.5 text-gray-700 text-sm">비밀번호 확인</label>
                  <div className="relative">
                    <input
                      type={showConfirmPassword ? "text" : "password"}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="비밀번호를 다시 입력해주세요"
                      className="w-full px-3.5 pr-11 py-2.5 border-2 border-blue-100 rounded-2xl focus:outline-none focus:border-blue-400 text-sm"
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

              <div className="bg-blue-50 border border-blue-200 rounded-2xl p-3 mb-4">
                <p className="text-xs text-gray-700 mb-1.5">비밀번호 조건:</p>
                <ul className="text-xs text-gray-600 space-y-0.5">
                  <li>• 6~16자</li>
                  <li>• 영문, 숫자, 특수문자(!?~@#$%&^) 중 2종류 이상</li>
                  <li>• 연속되거나 동일한 문자/숫자 불가</li>
                </ul>
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-2xl mb-4">
                  <AlertCircle
                    size={18}
                    className="text-red-500 flex-shrink-0"
                  />
                  <p className="text-xs text-red-700 leading-relaxed">{error}</p>
                </div>
              )}

              <div className="flex gap-2.5">
                <button
                  onClick={() => setStep(1)}
                  className="flex-1 py-3 border-2 border-blue-200 rounded-2xl hover:bg-blue-50 transition-all text-sm"
                >
                  이전
                </button>
                <button
                  onClick={handleCompleteReset}
                  disabled={!password || !confirmPassword || isResetting}
                  className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  {isResetting ? '처리 중...' : '비밀번호 재설정 완료'}
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Code Sent Popup */}
      {showCodeSentPopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Mail size={32} className="text-white" />
              </div>
            </div>
            <h2 className="text-xl text-center mb-2 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              인증 코드 발송 완료
            </h2>
            <p className="text-center text-gray-600 text-sm mb-4">
              이메일로 인증번호가 발송되었습니다.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4 mb-6 text-center text-sm text-gray-700">
              입력하신 이메일로 인증번호를 보냈습니다. 3분 이내에 인증을 완료해주세요.
            </div>
            <button
              onClick={() => setShowCodeSentPopup(false)}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
            >
              확인
            </button>
          </div>
        </div>
      )}

      {/* Reset Complete Popup */}
      {showResetCompletePopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Check size={32} className="text-white" />
              </div>
            </div>
            <h2 className="text-xl text-center mb-2 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              비밀번호 재설정 완료
            </h2>
            <p className="text-center text-gray-600 text-sm mb-6">
              비밀번호가 성공적으로 재설정되었습니다!<br />
              로그인하여 서비스를 이용해보세요.
            </p>
            <button
              onClick={() => {
                setShowResetCompletePopup(false);
                onNavigate('login');
              }}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
            >
              로그인하러 가기
            </button>
          </div>
        </div>
      )}

    </div>
  );
}