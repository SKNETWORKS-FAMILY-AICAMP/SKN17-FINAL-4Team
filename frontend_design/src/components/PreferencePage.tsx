import React, { useState, useEffect, useCallback } from 'react';
import { Check, Lamp, Loader2, AlertCircle } from 'lucide-react';
import type { Page } from '../types/navigation';
import { api, initCsrfToken } from '../lib/api';

interface PreferencePageProps {
  onNavigate: (page: Page) => void;
  onComplete: (preferences: any) => void;
  onLogout: () => void;
  initialPreferences?: {
    gender?: string | null;
    birthdate?: string | null;
    mbti?: string | null;
    styles?: string[];
    preferred_moods?: string[];
  } | null;
}

const styleImages = [
  { id: 'vintage', name: 'Vintage', displayName: '빈티지', url: 'https://images.unsplash.com/photo-1577176434922-803273eba97a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx2aW50YWdlJTIwZnVybml0dXJlfGVufDF8fHx8MTc2NDI4OTIwOXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'luxury', name: 'Luxury', displayName: '럭셔리', url: 'https://images.unsplash.com/photo-1562438668-bcf0ca6578f0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBiZWRyb29tfGVufDF8fHx8MTc2NDI2ODY5OHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'natural', name: 'Natural', displayName: '내추럴', url: 'https://images.unsplash.com/photo-1485819196298-11222a657b31?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuYXR1cmFsJTIwaG9tZSUyMGRlY29yfGVufDF8fHx8MTc2NDMwNzczNXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'scandinavian', name: 'Scandinavian', displayName: '스칸디나비안', url: 'https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzY2FuZGluYXZpYW4lMjBsaXZpbmclMjByb29tfGVufDF8fHx8MTc2NDIxMzE5N3ww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'french', name: 'French', displayName: '프렌치', url: 'https://images.unsplash.com/photo-1679586493118-dd400758aeab?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVuY2glMjBhcGFydG1lbnR8ZW58MXx8fHwxNzY0MzA3NzM2fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'lovely', name: 'Lovely', displayName: '러블리', url: 'https://images.unsplash.com/photo-1697162103256-dad37889d979?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjdXRlJTIwcGluayUyMGJlZHJvb218ZW58MXx8fHwxNzY0MzA3NzM2fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'pastel', name: 'Pastel', displayName: '파스텔', url: 'https://images.unsplash.com/photo-1592782423007-db7afd35017f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXN0ZWwlMjBiZWRyb29tfGVufDF8fHx8MTc2NDMwNzczNnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'modern', name: 'Modern', displayName: '모던', url: 'https://images.unsplash.com/photo-1631679706909-1844bbd07221?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBsaXZpbmclMjByb29tfGVufDF8fHx8MTc2NDI5NzIzMnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'bohemian', name: 'Bohemian', displayName: '보헤미안', url: 'https://images.unsplash.com/photo-1628304457639-3dd4ded74a8b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib2hlbWlhbiUyMGJlZHJvb218ZW58MXx8fHwxNzY0MzA3NzM3fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'classic', name: 'Classic', displayName: '클래식', url: 'https://images.unsplash.com/photo-1705299493246-9312166c7d18?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGFzc2ljJTIwZnVybml0dXJlJTIwcm9vbXxlbnwxfHx8fDE3NjQzMDc3Mzd8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'industrial', name: 'Industrial', displayName: '인더스트리얼', url: 'https://images.unsplash.com/photo-1758887250669-9ce43c44611d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbG9mdCUyMGFwYXJ0bWVudHxlbnwxfHx8fDE3NjQyNDM0MDN8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' },
  { id: 'minimal', name: 'Minimal', displayName: '미니멀', url: 'https://images.unsplash.com/photo-1610123172705-a57f116cd4d9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsaXN0JTIwYXBhcnRtZW50fGVufDF8fHx8MTc2NDI0NjI3MHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral' }
];

export function PreferencePage({ onNavigate, onComplete, onLogout, initialPreferences }: PreferencePageProps) {
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [gender, setGender] = useState('');
  const [birthdate, setBirthdate] = useState('');
  const [birthdateError, setBirthdateError] = useState('');
  const [mbti, setMbti] = useState<string[]>([]);
  const [selectedStyles, setSelectedStyles] = useState<string[]>([]);
  const [submissionError, setSubmissionError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [initializing, setInitializing] = useState(!initialPreferences);
  const [initialError, setInitialError] = useState('');

  const hydrateFromPreferences = useCallback((prefs: PreferencePageProps['initialPreferences']) => {
    if (!prefs) return;
    setGender(prefs.gender ?? '');
    if (prefs.birthdate) {
      setBirthdate(prefs.birthdate.replace(/-/g, ''));
    }
    if (prefs.mbti) {
      setMbti(prefs.mbti.split(''));
    }
    const moods = prefs.preferred_moods ?? prefs.styles ?? [];
    setSelectedStyles(moods);
    setInitializing(false);
  }, []);

  const mbtiOptions = [
    ['E', 'I'],
    ['S', 'N'],
    ['T', 'F'],
    ['J', 'P'],
  ];

  const fetchInitialData = useCallback(async () => {
    if (initialPreferences) {
      hydrateFromPreferences(initialPreferences);
      return;
    }
    setInitializing(true);
    setInitialError('');
    try {
      const [profileRes, prefsRes] = await Promise.all([
        api.get('/accounts/profile/'),
        api.get('/favorites/preferences/'),
      ]);
      hydrateFromPreferences({
        gender: profileRes.data.gender,
        birthdate: profileRes.data.birth_date,
        mbti: profileRes.data.mbti,
        preferred_moods: prefsRes.data?.preferred_moods ?? [],
      });
    } catch (error) {
      console.error('선호도 초기 데이터를 불러오지 못했습니다.', error);
      setInitialError('선호도 정보를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.');
      setInitializing(false);
    }
  }, [hydrateFromPreferences, initialPreferences]);

  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  const formatBirthdateToISO = (value: string) => {
    if (value.length !== 8) return null;
    return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}`;
  };

  const handleMbtiSelect = (index: number, value: string) => {
    const newMbti = [...mbti];
    newMbti[index] = value;
    setMbti(newMbti);
  };

  const toggleStyleSelection = (styleId: string) => {
    if (selectedStyles.includes(styleId)) {
      setSelectedStyles(selectedStyles.filter(id => id !== styleId));
    } else {
      if (selectedStyles.length < 3) {
        setSelectedStyles([...selectedStyles, styleId]);
      }
    }
  };

  const validateBirthdate = (value: string) => {
    // 숫자만 입력되었는지 확인
    if (!/^\d{8}$/.test(value)) {
      return false;
    }

    const year = parseInt(value.substring(0, 4));
    const month = parseInt(value.substring(4, 6));
    const day = parseInt(value.substring(6, 8));

    // 월 검증 (01-12)
    if (month < 1 || month > 12) {
      return false;
    }

    // 일 검증 (01-31)
    if (day < 1 || day > 31) {
      return false;
    }

    // 날짜가 유효한지 확인
    const date = new Date(year, month - 1, day);
    if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) {
      return false;
    }

    // 오늘 날짜보다 이후인지 확인 (한국 표준시)
    const today = new Date();
    const kstToday = new Date(today.getTime() + 9 * 60 * 60 * 1000); // UTC+9
    const kstTodayString = kstToday.toISOString().slice(0, 10).replace(/-/g, '');

    if (parseInt(value) > parseInt(kstTodayString)) {
      return false;
    }

    return true;
  };

  const handleBirthdateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    
    // 숫자가 아닌 문자가 입력되면 제거
    const numbersOnly = value.replace(/[^0-9]/g, '');
    setBirthdate(numbersOnly);

    // 입력이 시작되면 일단 에러 메시지 표시
    if (numbersOnly.length > 0 && numbersOnly.length < 8) {
      setBirthdateError('생년월일을 올바른 형식으로 입력해 주시기 바랍니다.');
      return;
    }

    // 입력이 없으면 에러 메시지 제거
    if (numbersOnly.length === 0) {
      setBirthdateError('');
      return;
    }

    // 8자리 입력 완료 시 검증
    if (numbersOnly.length === 8) {
      if (!validateBirthdate(numbersOnly)) {
        setBirthdateError('생년월일을 다시 입력해 주시기 바랍니다.');
      } else {
        setBirthdateError('');
      }
    }
  };

  const handleNext = () => {
    if (step === 1 && gender) {
      setStep(2);
    } else if (step === 2 && birthdate) {
      setStep(3);
    } else if (step === 3 && mbti.length === 4) {
      setStep(4);
    } else if (step === 4 && selectedStyles.length >= 1) {
      submitPreferences();
    }
  };

  const submitPreferences = async () => {
    setSubmissionError('');
    const birthdateIso = birthdate ? formatBirthdateToISO(birthdate) : null;
    try {
      setIsSubmitting(true);
      await initCsrfToken();
      await Promise.all([
        api.patch('/accounts/profile/', {
          gender: gender || null,
          birth_date: birthdateIso,
          mbti: mbti.join('') || null,
        }),
        api.post('/favorites/preferences/', {
          preferred_moods: selectedStyles,
        }),
      ]);
      onComplete({
        gender,
        birthdate: birthdateIso,
        mbti: mbti.join(''),
        styles: selectedStyles,
      });
    } catch (error) {
      console.error('선호도 저장 실패', error);
      setSubmissionError('선호도를 저장하지 못했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const canProceed = () => {
    if (step === 1) return gender !== '';
    if (step === 2) return birthdate !== '' && validateBirthdate(birthdate) && birthdateError === '';
    if (step === 3) return mbti.length === 4;
    if (step === 4) return selectedStyles.length >= 1 && selectedStyles.length <= 3;
    return false;
  };

  if (initializing) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 via-white to-yellow-50">
        <Loader2 size={48} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (initialError) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-gradient-to-br from-blue-50 via-white to-yellow-50 px-4 text-center">
        <AlertCircle size={48} className="text-red-500" />
        <p className="text-lg text-gray-700">{initialError}</p>
        <button
          onClick={fetchInitialData}
          className="rounded-full bg-blue-500 px-6 py-3 text-white shadow hover:bg-blue-600"
        >
          다시 시도하기
        </button>
      </div>
    );
  }

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
          <button
            onClick={() => onNavigate('reference')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            레퍼런스 보드
          </button>
          <button
            onClick={() => onNavigate('preference')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            취향분석
          </button>
          <button
            onClick={onLogout}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            로그아웃
          </button>
        </div>
      </div>

      <main className="pt-32 pb-16">
        <div className="max-w-4xl mx-auto px-6">
          {/* Step 1: Gender */}
          {step === 1 && (
            <div className="space-y-12">
              <h1 className="text-4xl mb-16 text-center">당신에 대해 알려주세요!</h1>

              <div>
                <h2 className="text-xl mb-6 text-center">당신의 성별은 무엇인가요?</h2>
                <div className="w-full max-w-[520px] mx-auto">
                  <div className="flex gap-4">
                    <button
                      onClick={() => setGender('여성')}
                      className={`w-full py-3 rounded-full border-2 transition-colors ${
                        gender === '여성'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      여성
                    </button>
                    <button
                      onClick={() => setGender('남성')}
                      className={`w-full py-3 rounded-full border-2 transition-colors ${
                        gender === '남성'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      남성
                    </button>
                    <button
                      onClick={() => setGender('선택 안함')}
                      className={`w-full py-3 rounded-full border-2 transition-colors ${
                        gender === '선택 안함'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      선택 안함
                    </button>
                  </div>
                </div>
              </div>

              <div className="w-full max-w-[520px] mx-auto">
                <div className="pt-8 flex gap-3">
                  <button
                    onClick={() => onNavigate('chat')}
                    className="w-full py-3 border-2 border-blue-500 text-blue-600 rounded-full hover:bg-blue-50 transition-colors"
                  >
                    채팅으로
                  </button>
                  <button
                    onClick={handleNext}
                    disabled={!canProceed()}
                    className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    다음 단계 &gt;
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Birthdate */}
          {step === 2 && (
            <div className="space-y-12">
              <h1 className="text-4xl mb-16 text-center">당신에 대해 알려주세요!</h1>

              <div>
                <h2 className="text-xl mb-6 text-center">생년월일을 알려주세요.</h2>
                <p className="text-sm text-gray-500 mb-4 text-center">
                  YYYYMMDD 형식으로 8자리 숫자를 입력해주세요
                </p>
                <div className="flex justify-center">
                  <input
                    type="text"
                    value={birthdate}
                    onChange={handleBirthdateChange}
                    placeholder="예: 19900101"
                    maxLength={8}
                    className="w-[520px] h-[52.2px] px-6 border-2 border-blue-200 rounded-full focus:outline-none focus:border-blue-400 text-center"
                  />
                </div>
                {birthdateError && (
                  <p className="text-red-500 text-sm mt-2 text-center">{birthdateError}</p>
                )}
              </div>

              <div className="w-full max-w-[520px] mx-auto">
                <div className="pt-8 flex gap-3">
                  <button
                    onClick={() => setStep(1)}
                    className="w-full py-3 border-2 border-blue-500 text-blue-600 rounded-full hover:bg-blue-50 transition-colors"
                  >
                    이전
                  </button>
                  <button
                    onClick={handleNext}
                    disabled={!canProceed()}
                    className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    다음 단계 &gt;
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: MBTI */}
          {step === 3 && (
            <div className="space-y-12">
              <h1 className="text-4xl mb-16 text-center">당신에 대해 알려주세요!</h1>

              <div className="w-full max-w-[520px] mx-auto">
                <h2 className="text-xl mb-6 text-center">당신의 MBTI는 어떤 타입인가요?</h2>
                <div className="space-y-4">
                  {/* First row: E, S, T, J */}
                  <div className="flex gap-4 justify-center">
                    <button
                      onClick={() => handleMbtiSelect(0, 'E')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[0] === 'E'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      E
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(1, 'S')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[1] === 'S'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      S
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(2, 'T')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[2] === 'T'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      T
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(3, 'J')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[3] === 'J'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      J
                    </button>
                  </div>

                  {/* Second row: I, N, F, P */}
                  <div className="flex gap-4 justify-center">
                    <button
                      onClick={() => handleMbtiSelect(0, 'I')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[0] === 'I'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      I
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(1, 'N')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[1] === 'N'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      N
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(2, 'F')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[2] === 'F'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      F
                    </button>
                    <button
                      onClick={() => handleMbtiSelect(3, 'P')}
                      className={`w-full px-8 py-3 rounded-full border-2 transition-colors ${
                        mbti[3] === 'P'
                          ? 'bg-gradient-to-r from-blue-500 to-blue-400 text-white border-blue-500 shadow-lg'
                          : 'border-blue-200 hover:border-blue-400'
                      }`}
                    >
                      P
                    </button>
                  </div>
                </div>
              </div>

              <div className="w-full max-w-[520px] mx-auto">
                <div className="pt-8 flex gap-3">
                  <button
                    onClick={() => setStep(2)}
                    className="w-full py-3 border-2 border-blue-500 text-blue-600 rounded-full hover:bg-blue-50 transition-colors"
                  >
                    이전
                  </button>
                  <button
                    onClick={handleNext}
                    disabled={!canProceed()}
                    className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    다음 단계 &gt;
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Style Selection */}
          {step === 4 && (
            <div className="space-y-5">
              <div className="text-center">
                <h1 className="text-4xl mb-4">당신의 취향을 알려주세요!</h1>
                <p className="text-gray-600">마음에 드는 사진을 1개~3개까지 골라주세요 :)</p>
              </div>

              <div className="w-full max-w-[520px] mx-auto">
                <div className="grid grid-cols-4 gap-3">
                  {styleImages.map(style => (
                    <button
                      key={style.id}
                      onClick={() => toggleStyleSelection(style.id)}
                      className="relative aspect-square rounded-xl overflow-hidden group"
                    >
                      <img
                        src={style.url}
                        alt={style.name}
                        className="w-full h-full object-cover"
                      />

                      {/* Hover overlay with category name */}
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-60 transition-all duration-300 flex items-center justify-center">
                        <p className="text-white text-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                          {style.displayName}
                        </p>
                      </div>

                      {/* Selection indicator */}
                      {selectedStyles.includes(style.id) && (
                        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                          <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                            <Check size={20} strokeWidth={3} />
                          </div>
                        </div>
                      )}
                    </button>
                  ))}
                </div>

                <div className="pt-6 flex gap-3">
                  <button
                    onClick={() => setStep(3)}
                    className="w-full py-3 border-2 border-blue-500 text-blue-600 rounded-full hover:bg-blue-50 transition-colors"
                  >
                    이전
                  </button>
                  {submissionError && (
                    <p className="flex-1 text-center text-sm text-red-500">{submissionError}</p>
                  )}
                  <button
                    onClick={handleNext}
                    disabled={!canProceed() || isSubmitting}
                    className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-full hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    {isSubmitting ? '저장 중...' : '완료하기'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}