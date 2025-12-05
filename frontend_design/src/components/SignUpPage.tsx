import React, { useState, useEffect } from 'react';
import { Check, Mail, Lock, AlertCircle, Clock, Eye, EyeOff, Lamp } from 'lucide-react';
import { api, initCsrfToken } from '../lib/api';
import type { Page } from '../types/navigation';

interface SignUpPageProps {
  onNavigate: (page: Page) => void;
  onSignUp: () => void;
}

export function SignUpPage({ onNavigate, onSignUp }: SignUpPageProps) {
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [email, setEmail] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [privacyAccepted, setPrivacyAccepted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  const [sendCount, setSendCount] = useState(0);
  const [error, setError] = useState('');
  const [showTermsPopup, setShowTermsPopup] = useState(false);
  const [showPrivacyPopup, setShowPrivacyPopup] = useState(false);
  const [showCodeSentPopup, setShowCodeSentPopup] = useState(false);
  const [showSignUpCompletePopup, setShowSignUpCompletePopup] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [codeRequested, setCodeRequested] = useState(false);
  const [isSendingCode, setIsSendingCode] = useState(false);
  const [isVerifyingCode, setIsVerifyingCode] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 타이머
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
      await api.post('/accounts/register/email/', { email });
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
      setError('먼저 인증번호를 발송해주세요.');
      return;
    }

    if (timeLeft === 0) {
      setError('인증 시간이 만료되었습니다. 인증번호를 재발송해주세요.');
      return;
    }

    try {
      setIsVerifyingCode(true);
      await initCsrfToken();
      await api.post('/accounts/register/verify/', {
        email,
        code: verificationCode,
      });
      setStep(3);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsVerifyingCode(false);
    }
  };

  const handleCompleteSignUp = async () => {
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
      setIsSubmitting(true);
      await initCsrfToken();
      await api.post('/accounts/register/complete/', {
        email,
        password,
        password2: confirmPassword,
      });
      setShowSignUpCompletePopup(true);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
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

      {/* ✅ 여기 레이아웃만 수정: items-center → items-start, py-4 → pt-12 pb-6 */}
      <main className="h-[calc(100vh-80px)] flex justify-center items-start pt-30 pb-6">
        <div className="max-w-md mx-auto px-6 w-full">
          {/* Progress - 카드 바깥, 항상 같은 위치 */}
          <div className="flex items-center justify-center mb-6">
            {[1, 2, 3, 4].map((s) => (
              <React.Fragment key={s}>
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center transition-all text-sm ${
                    step >= s
                      ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white shadow-lg'
                      : 'bg-gray-200 text-gray-400'
                  }`}
                >
                  {step > s ? <Check size={16} /> : s}
                </div>
                {s < 4 && (
                  <div
                    className={`w-12 h-1 mx-1.5 transition-all ${
                      step > s
                        ? 'bg-gradient-to-r from-blue-500 to-blue-400'
                        : 'bg-gray-200'
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Step 1: 약관 동의 */}
          {step === 1 && (
            <div className="bg-white rounded-3xl shadow-xl p-6">
              <h1 className="text-2xl mb-1 text-center bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                약관 동의
              </h1>
              <p className="text-gray-500 text-center mb-5 text-sm">
                서비스 이용을 위해 약관에 동의해주세요
              </p>

              <div className="space-y-3 mb-5">
                <label className="flex items-start gap-2.5 p-3 border-2 border-blue-100 rounded-2xl hover:border-blue-300 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={termsAccepted}
                    onChange={(e) => setTermsAccepted(e.target.checked)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-0.5">
                      <p className="text-sm">서비스 이용약관 (필수)</p>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          setShowTermsPopup(true);
                        }}
                        className="text-xs text-blue-600 hover:text-blue-700 underline"
                      >
                        내용보기
                      </button>
                    </div>
                    <p className="text-xs text-gray-500">
                      MOOD ON 서비스 이용을 위한 약관입니다.
                    </p>
                  </div>
                </label>

                <label className="flex items-start gap-2.5 p-3 border-2 border-blue-100 rounded-2xl hover:border-blue-300 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={privacyAccepted}
                    onChange={(e) => setPrivacyAccepted(e.target.checked)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-0.5">
                      <p className="text-sm">개인정보 처리방침 (필수)</p>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          setShowPrivacyPopup(true);
                        }}
                        className="text-xs text-blue-600 hover:text-blue-700 underline"
                      >
                        내용보기
                      </button>
                    </div>
                    <p className="text-xs text-gray-500">
                      개인정보 수집 및 이용에 대한 안내입니다.
                    </p>
                  </div>
                </label>
              </div>

              <button
                onClick={() => setStep(2)}
                disabled={!termsAccepted || !privacyAccepted}
                className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                다음
              </button>
            </div>
          )}

          {/* Step 2: 이메일 인증 */}
          {step === 2 && (
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

              <div className="flex gap-2.5">
                <button
                  onClick={() => setStep(1)}
                  className="flex-1 py-3 border-2 border-blue-200 rounded-2xl hover:bg-blue-50 transition-all text-sm"
                >
                  이전
                </button>
                <button
                  onClick={handleVerifyCode}
                  disabled={!verificationCode || !codeRequested || isVerifyingCode}
                  className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  {isVerifyingCode ? '인증 중...' : '인증하기'}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: 비밀번호 설정 */}
          {step === 3 && (
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
                  onClick={() => setStep(2)}
                  className="flex-1 py-3 border-2 border-blue-200 rounded-2xl hover:bg-blue-50 transition-all text-sm"
                >
                  이전
                </button>
                <button
                  onClick={handleCompleteSignUp}
                  disabled={!password || !confirmPassword || isSubmitting}
                  className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  {isSubmitting ? '처리 중...' : '회원가입 완료'}
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
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4 mb-6">
              <p className="text-center text-sm text-gray-700">
                입력하신 이메일로 인증번호를 보냈습니다. 3분 이내에 인증을 완료해주세요.
              </p>
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

      {/* Sign Up Complete Popup */}
      {showSignUpCompletePopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Check size={32} className="text-white" />
              </div>
            </div>
            <h2 className="text-xl text-center mb-2 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              회원가입 완료
            </h2>
            <p className="text-center text-gray-600 text-sm mb-6">
              MOOD ON 회원가입이 완료되었습니다!<br />
              로그인하여 서비스를 이용해보세요.
            </p>
            <button
              onClick={() => {
                setShowSignUpCompletePopup(false);
                onNavigate('login');
              }}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
            >
              로그인하러 가기
            </button>
          </div>
        </div>
      )}

      {/* Terms Popup */}
      {showTermsPopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-6 max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Check size={32} className="text-white" />
              </div>
            </div>
            <h2 className="text-xl text-center mb-2 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              서비스 이용약관
            </h2>
            <p className="text-center text-gray-500 text-sm mb-4">
              약관을 확인하시려면 아래 내용을 읽어주세요.
            </p>
            <div className="flex-1 overflow-y-auto space-y-4 text-sm text-gray-700 mb-6">
              <section>
                <h3 className="mb-2 text-blue-600">제1조 (목적)</h3>
                <p>
                  이 약관은 MOOD ON(이하 "회사")이 제공하는 인테리어 상품 추천 서비스 이용과
                  관련하여 회사와 이용자의 권리, 의무 및 책임사항, 기타 필요한 사항을 규정함을
                  목적으로 합니다.
                </p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제2조 (정의)</h3>
                <p>
                  1. "서비스"라 함은 회사가 제공하는 AI 기반 인테리어 상품 추천, 챗봇 상담, 스타일
                  분석 등 모든 서비스를 의미합니다.
                </p>
                <p>2. "이용자"라 함은 본 약관에 따라 회사가 제공하는 서비스를 받는 회원을 말합니다.</p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제3조 (서비스의 제공)</h3>
                <p>1. 회사는 이용자에게 AI 챗봇을 통한 인테리어 상품 추천 서비스를 제공합니다.</p>
                <p>
                  2. 서비스는 연중무휴 1일 24시간 제공함을 원칙으로 합니다. 다만, 시스템 정기점검 등의
                  사유로 서비스가 일시 중단될 수 있습니다.
                </p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제4조 (이용자의 의무)</h3>
                <p>1. 이용자는 서비스 이용 시 다음 각 호의 행위를 하여서는 안 됩니다:</p>
                <ul className="list-disc ml-6 mt-2 space-y-1">
                  <li>타인의 개인정보를 도용하는 행위</li>
                  <li>불법적이거나 부적절한 내용을 게시하는 행위</li>
                  <li>서비스의 안정적 운영을 방해하는 행위</li>
                </ul>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제5조 (AI 추천의 한계)</h3>
                <p>1. AI 추천은 참고용이며, 최종 구매 결정은 이용자의 책임입니다.</p>
                <p>2. 상품의 재고, 가격, 배송 정보는 실시간과 다를 수 있습니다.</p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제6조 (면책조항)</h3>
                <p>
                  회사는 천재지변, 전쟁, 기타 이에 준하는 불가항력으로 인하여 서비스를 제공할 수 없는
                  경우에는 서비스 제공에 관한 책임이 면제됩니다.
                </p>
              </section>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setShowTermsPopup(false)}
                className="flex-1 py-3 bg-white border-2 border-gray-200 text-gray-700 rounded-2xl hover:bg-gray-50 transition-all"
              >
                취소
              </button>
              <button
                onClick={() => setShowTermsPopup(false)}
                className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Privacy Popup */}
      {showPrivacyPopup && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl p-6 max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <Lock size={32} className="text-white" />
              </div>
            </div>
            <h2 className="text-xl text-center mb-2 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              개인정보 처리방침
            </h2>
            <p className="text-center text-gray-500 text-sm mb-4">
              개인정보 처리방침을 확인하시려면 아래 내용을 읽어주세요.
            </p>
            <div className="flex-1 overflow-y-auto space-y-4 text-sm text-gray-700 mb-6">
              <section>
                <h3 className="mb-2 text-blue-600">제1조 (개인정보의 수집 및 이용 목적)</h3>
                <p>회사는 다음의 목적을 위하여 개인정보를 처리합니다:</p>
                <ul className="list-disc ml-6 mt-2 space-y-1">
                  <li>회원가입 및 관리</li>
                  <li>서비스 제공 및 맞춤형 추천</li>
                  <li>서비스 개선 및 통계 분석</li>
                </ul>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제2조 (수집하는 개인정보의 항목)</h3>
                <p>회사는 다음의 개인정보 항목을 수집합니다:</p>
                <ul className="list-disc ml-6 mt-2 space-y-1">
                  <li>필수항목: 이메일, 비밀번호</li>
                  <li>선택항목: 성별, 생년월일, MBTI, 선호 스타일</li>
                  <li>자동 수집: IP주소, 쿠키, 서비스 이용 기록</li>
                </ul>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제3조 (개인정보의 보유 및 이용기간)</h3>
                <p>
                  회사는 법령에 따른 개인정보 보유·이용기간 또는 정보주체로부터 개인정보를 수집 시에
                  동의받은 개인정보 보유·이용기간 내에서 개인정보를 처리·보유합니다.
                </p>
                <p className="mt-2">
                  회원 탈퇴 시 즉시 파기됩니다. (단, 관련 법령에 의거하여 보존할 필요가 있는 경우 해당
                  기간 동안 보관)
                </p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제4조 (개인정보의 제3자 제공)</h3>
                <p>
                  회사는 원칙적으로 이용자의 개인정보를 제3자에게 제공하지 않습니다. 다만, 이용자의 동의가
                  있거나 법령의 규정에 의한 경우는 예외로 합니다.
                </p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제5조 (개인정보의 파기)</h3>
                <p>
                  회사는 개인정보 보유기간의 경과, 처리목적 달성 등 개인정보가 불필요하게 되었을 때에는
                  지체없이 해당 개인정보를 파기합니다.
                </p>
              </section>
              <section>
                <h3 className="mb-2 text-blue-600">제6조 (이용자의 권리·의무)</h3>
                <p>
                  이용자는 언제든지 등록되어 있는 자신의 개인정보를 조회하거나 수정할 수 있으며, 가입해지를
                  요청할 수 있습니다.
                </p>
              </section>
              <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-4 mt-6">
                <p className="text-sm text-yellow-800">
                  ⚠️ 주의: MOOD ON은 PII(개인식별정보) 수집을 최소화합니다. 주민등록번호, 전화번호,
                  정확한 주소 등 민감한 정보를 입력하지 마세요.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setShowPrivacyPopup(false)}
                className="flex-1 py-3 bg-white border-2 border-gray-200 text-gray-700 rounded-2xl hover:bg-gray-50 transition-all"
              >
                취소
              </button>
              <button
                onClick={() => setShowPrivacyPopup(false)}
                className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}