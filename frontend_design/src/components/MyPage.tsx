import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Heart,
  ExternalLink,
  AlertCircle,
  ArrowUpDown,
  User,
  Calendar,
  Sparkles,
  Palette,
  Lamp,
  Loader2,
} from 'lucide-react';
import type { Page } from '../types/navigation';
import { api } from '../lib/api';

type FavoriteProductRecord = {
  id: number;
  product?: {
    id: number;
    product_name: string;
    brand_name: string;
    image_url: string;
    link_url: string;
    price: number;
  };
  created_at: string;
};

type SortOption = 'recent' | 'price-low' | 'price-high';

interface MyPageProps {
  onNavigate: (page: Page) => void;
  onLogout: () => void;
  userEmail: string;
  onDeleteAccount: (payload: { password: string }) => Promise<void>;
  initialPreferences?: {
    gender?: string | null;
    birthdate?: string | null;
    mbti?: string | null;
    preferred_moods?: string[];
    styles?: string[];
  } | null;
  initialFavorites?: FavoriteProductRecord[];
  onRefreshUserData?: () => Promise<void> | void;
}

const MOOD_OPTIONS = [
  { id: 'vintage', label: '빈티지' },
  { id: 'luxury', label: '럭셔리' },
  { id: 'natural', label: '내추럴' },
  { id: 'scandinavian', label: '스칸디' },
  { id: 'french', label: '프렌치' },
  { id: 'lovely', label: '러블리' },
  { id: 'pastel', label: '파스텔' },
  { id: 'modern', label: '모던' },
  { id: 'bohemian', label: '보헤미안' },
  { id: 'classic', label: '클래식' },
  { id: 'industrial', label: '인더스트리얼' },
  { id: 'minimal', label: '미니멀' },
];

const formatPrice = (price?: number) => {
  if (typeof price !== 'number' || Number.isNaN(price)) return '가격 정보 없음';
  return `₩${price.toLocaleString()}`;
};

const normalizeBirth = (value?: string | null) => value ?? '';

export function MyPage({
  onNavigate,
  onLogout,
  userEmail,
  onDeleteAccount,
  initialPreferences,
  initialFavorites,
  onRefreshUserData,
}: MyPageProps) {
  const [showDeleteAccountPopup, setShowDeleteAccountPopup] = useState(false);
  const [sortBy, setSortBy] = useState<SortOption>('recent');
  const [showSortMenu, setShowSortMenu] = useState(false);
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  const [profile, setProfile] = useState({
    gender: initialPreferences?.gender ?? '',
    birth_date: normalizeBirth(initialPreferences?.birthdate),
    mbti: initialPreferences?.mbti ?? '',
  });
  const [preferredMoods, setPreferredMoods] = useState<string[]>(
    initialPreferences?.preferred_moods ?? initialPreferences?.styles ?? [],
  );
  const [favorites, setFavorites] = useState<FavoriteProductRecord[]>(initialFavorites ?? []);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [profileSaving, setProfileSaving] = useState(false);
  const [prefsSaving, setPrefsSaving] = useState(false);
  const [favoriteRemovingId, setFavoriteRemovingId] = useState<number | null>(null);
  const [toastMessage, setToastMessage] = useState('');

  const displayEmail = userEmail || 'user@moodon.com';
  const displayName = displayEmail.split('@')[0] || 'MoodOn 사용자';

  const displayGender = profile.gender || '미입력';
  const displayBirthdate = profile.birth_date ? profile.birth_date.replace(/-/g, '.') : '정보 없음';
  const displayMbti = profile.mbti || '미입력';
  const displayStyles = preferredMoods.length ? preferredMoods.join(', ') : '선호 스타일 없음';

  const showToast = (message: string) => {
    setToastMessage(message);
    setTimeout(() => setToastMessage(''), 3000);
  };

  const fetchMyPageData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [profileRes, preferenceRes, favoritesRes] = await Promise.all([
        api.get('/accounts/profile/'),
        api.get('/favorites/preferences/'),
        api.get('/favorites/'),
      ]);
      const moods = preferenceRes.data?.preferred_moods ?? [];
      setProfile({
        gender: profileRes.data.gender ?? '',
        birth_date: normalizeBirth(profileRes.data.birth_date),
        mbti: profileRes.data.mbti ?? '',
      });
      setPreferredMoods(moods);
      setFavorites(favoritesRes.data ?? []);
      await onRefreshUserData?.();
    } catch (err) {
      console.error('마이페이지 데이터를 불러오지 못했습니다.', err);
      setError('데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  }, [onRefreshUserData]);

  useEffect(() => {
    fetchMyPageData();
  }, [fetchMyPageData]);

  useEffect(() => {
    if (initialPreferences) {
      setProfile({
        gender: initialPreferences.gender ?? '',
        birth_date: normalizeBirth(initialPreferences.birthdate),
        mbti: initialPreferences.mbti ?? '',
      });
      setPreferredMoods(initialPreferences.preferred_moods ?? initialPreferences.styles ?? []);
    }
  }, [initialPreferences]);

  useEffect(() => {
    if (initialFavorites) {
      setFavorites(initialFavorites);
    }
  }, [initialFavorites]);

  const sortedProducts = useMemo(() => {
    const sorted = [...favorites];
    if (sortBy === 'price-low') {
      return sorted.sort((a, b) => {
        const priceA = a.product?.price ?? 0;
        const priceB = b.product?.price ?? 0;
        return priceA - priceB;
      });
    }
    if (sortBy === 'price-high') {
      return sorted.sort((a, b) => {
        const priceA = a.product?.price ?? 0;
        const priceB = b.product?.price ?? 0;
        return priceB - priceA;
      });
    }
    return sorted;
  }, [favorites, sortBy]);

  const getSortLabel = () => {
    switch (sortBy) {
      case 'price-low':
        return '가격 낮은순';
      case 'price-high':
        return '가격 높은순';
      default:
        return '등록순';
    }
  };

  const handleDeleteAccount = () => {
    setDeletePassword('');
    setDeleteError('');
    setShowDeleteAccountPopup(true);
  };

  const confirmDeleteAccount = async () => {
    if (!deletePassword) {
      setDeleteError('비밀번호를 입력해주세요.');
      return;
    }
    try {
      setIsDeleting(true);
      await onDeleteAccount({ password: deletePassword });
      setShowDeleteAccountPopup(false);
      setShowSuccessPopup(true);
    } catch (error) {
      setDeleteError('비밀번호가 올바르지 않습니다.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleSuccessConfirm = () => {
    setShowSuccessPopup(false);
    onLogout();
  };

  const handleProfileFieldChange = (field: 'gender' | 'birth_date' | 'mbti', value: string) => {
    setProfile(prev => ({
      ...prev,
      [field]: field === 'mbti' ? value.toUpperCase() : value,
    }));
  };

  const handleProfileSave = async () => {
    setProfileSaving(true);
    try {
      await api.patch('/accounts/profile/', {
        gender: profile.gender || null,
        birth_date: profile.birth_date || null,
        mbti: profile.mbti || null,
      });
      showToast('프로필 정보를 저장했습니다.');
      await onRefreshUserData?.();
    } catch (err) {
      console.error('프로필 저장 실패', err);
      setError('프로필 정보를 저장하지 못했습니다.');
    } finally {
      setProfileSaving(false);
    }
  };

  const toggleMood = (moodId: string) => {
    setPreferredMoods(prev => {
      if (prev.includes(moodId)) {
        return prev.filter(id => id !== moodId);
      }
      if (prev.length >= 3) {
        showToast('선호 무드는 최대 3개까지 선택할 수 있습니다.');
        return prev;
      }
      return [...prev, moodId];
    });
  };

  const handlePreferencesSave = async () => {
    if (preferredMoods.length === 0) {
      setError('최소 1개의 선호 무드를 선택해주세요.');
      return;
    }
    setPrefsSaving(true);
    try {
      await api.post('/favorites/preferences/', {
        preferred_moods: preferredMoods,
      });
      showToast('선호 무드를 저장했습니다.');
      await onRefreshUserData?.();
    } catch (err) {
      console.error('선호 무드 저장 실패', err);
      setError('선호 무드를 저장하지 못했습니다.');
    } finally {
      setPrefsSaving(false);
    }
  };

  const handleFavoriteDelete = async (favoriteId: number) => {
    if (!window.confirm('관심 상품 목록에서 삭제하시겠습니까?')) return;
    setFavoriteRemovingId(favoriteId);
    try {
      await api.delete(`/favorites/${favoriteId}/`);
      setFavorites(prev => prev.filter(item => item.id !== favoriteId));
      showToast('관심 상품을 삭제했습니다.');
    } catch (err) {
      console.error('관심 상품 삭제 실패', err);
      setError('관심 상품 삭제에 실패했습니다.');
    } finally {
      setFavoriteRemovingId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 via-white to-yellow-50">
        <Loader2 size={48} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error && favorites.length === 0 && preferredMoods.length === 0) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-gradient-to-br from-blue-50 via-white to-yellow-50 px-4 text-center">
        <AlertCircle size={48} className="text-red-500" />
        <p className="text-lg text-gray-700">{error}</p>
        <button
          onClick={fetchMyPageData}
          className="rounded-full bg-blue-500 px-6 py-3 text-white shadow hover:bg-blue-600"
        >
          다시 시도하기
        </button>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 via-white to-yellow-50 overflow-hidden">
      {/* Header */}
      <div className="border-b border-blue-100 px-5 py-3.5 flex items-center justify-between bg-white/80 backdrop-blur-sm shadow-sm fixed top-0 left-0 right-0 z-50">
        <div className="flex items-center gap-3">
          <button onClick={() => onNavigate('chat')} className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <div className="w-9 h-9 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center shadow-md">
              <Lamp size={18} className="text-white" />
            </div>
            <span className="text-[20px] font-medium leading-none bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent select-none">
              MOOD ON
            </span>
          </button>
        </div>
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

      <main className="pt-24 pb-2 h-full overflow-hidden">
        <div className="max-w-[1800px] mx-auto px-4 scale-[0.90] origin-top h-full">
          {error && (
            <div className="mb-4 flex items-start justify-between rounded-3xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <div className="flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5" />
                <span>{error}</span>
              </div>
              <button onClick={() => setError('')} className="text-xs underline">
                닫기
              </button>
            </div>
          )}
          <div className="grid grid-cols-[320px_1fr] gap-4 h-full overflow-hidden">
            {/* Left Column */}
            <div className="space-y-4 h-full overflow-y-auto">
              {/* Profile summary */}
              <div className="bg-white rounded-3xl p-6 shadow-md border-2 border-blue-100">
                <div className="text-center mb-5">
                  <div className="inline-block relative mb-4">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400 rounded-full flex items-center justify-center shadow-md">
                      <span className="text-3xl text-white">{displayName.slice(0, 1).toUpperCase()}</span>
                    </div>
                    <div className="absolute -bottom-1 -right-1 w-8 h-8 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full flex items-center justify-center shadow-md">
                      <Sparkles size={16} className="text-white" />
                    </div>
                  </div>
                  <h1 className="text-2xl mb-2 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                    {displayName}
                  </h1>
                  <p className="text-sm text-gray-600">{displayEmail}</p>
                </div>
                <div className="pt-5 border-t border-gray-100">
                  <h2 className="text-sm mb-3 flex items-center gap-2 text-gray-700">
                    <User size={18} className="text-blue-500" />
                    <span>계정 관리</span>
                  </h2>
                  <div className="flex gap-3">
                    <button
                      onClick={() => onNavigate('password-change')}
                      className="flex-1 text-left py-3 px-4 bg-gradient-to-r from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 rounded-2xl transition-all group flex items-center justify-between text-sm"
                    >
                      <span className="text-gray-800">비밀번호 수정</span>
                      <span className="text-blue-500 group-hover:translate-x-1 transition-transform">→</span>
                    </button>
                    <button
                      onClick={handleDeleteAccount}
                      className="flex-1 text-left py-3 px-4 bg-gradient-to-r from-red-50 to-pink-50 hover:from-red-100 hover:to-pink-100 rounded-2xl transition-all group flex items-center justify-between text-sm"
                    >
                      <span className="text-red-700">회원 탈퇴</span>
                      <span className="text-red-500 group-hover:translate-x-1 transition-transform">→</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Mood summary */}
              <div className="bg-white rounded-2xl p-3 shadow-md border-2 border-purple-100 flex-1 flex flex-col">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-6 h-6 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg flex items-center justify-center shadow-sm">
                    <Palette size={13} className="text-white" />
                  </div>
                  <h2 className="text-m bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent font-medium">
                    나의 MOOD
                  </h2>
                </div>
                <div className="grid grid-cols-2 gap-3 flex-1">
                  <InfoTile
                    label="성별"
                    value={displayGender}
                    icon={<User size={10} className="text-purple-500" />}
                    emoji={displayGender === '여성' ? '👩' : displayGender === '남성' ? '👨' : '🧑'}
                  />
                  <InfoTile
                    label="생년월일"
                    value={displayBirthdate}
                    icon={<Calendar size={10} className="text-pink-500" />}
                    emoji="🎂"
                  />
                  <InfoTile
                    label="MBTI"
                    value={displayMbti}
                    icon={<Sparkles size={10} className="text-green-500" />}
                    emoji={displayMbti}
                  />
                  <InfoTile
                    label="선호 무드"
                    value={displayStyles}
                    icon={<Palette size={10} className="text-blue-500" />}
                    emoji="🏠"
                  />
                </div>
              </div>

              {/* Profile form */}
              <div className="bg-white rounded-3xl p-5 shadow-md border border-blue-100 space-y-4">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                  <User size={18} className="text-blue-500" />
                  프로필 설정
                </h3>
                <div className="space-y-3 text-sm text-gray-600">
                  <label className="block">
                    <span className="mb-1 inline-block text-xs text-gray-500">성별</span>
                    <select
                      value={profile.gender}
                      onChange={(e) => handleProfileFieldChange('gender', e.target.value)}
                      className="w-full rounded-2xl border border-gray-200 px-4 py-2.5 focus:border-blue-400 focus:outline-none"
                    >
                      <option value="">선택 안함</option>
                      <option value="여성">여성</option>
                      <option value="남성">남성</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="mb-1 inline-block text-xs text-gray-500">생년월일</span>
                    <input
                      type="date"
                      value={profile.birth_date}
                      onChange={(e) => handleProfileFieldChange('birth_date', e.target.value)}
                      className="w-full rounded-2xl border border-gray-200 px-4 py-2.5 focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 inline-block text-xs text-gray-500">MBTI</span>
                    <input
                      maxLength={4}
                      value={profile.mbti}
                      onChange={(e) => handleProfileFieldChange('mbti', e.target.value.replace(/[^a-zA-Z]/g, '').toUpperCase())}
                      placeholder="예: ENFP"
                      className="w-full uppercase rounded-2xl border border-gray-200 px-4 py-2.5 focus:border-blue-400 focus:outline-none"
                    />
                  </label>
                  <button
                    onClick={handleProfileSave}
                    disabled={profileSaving}
                    className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg disabled:opacity-50"
                  >
                    {profileSaving ? '저장 중...' : '프로필 저장'}
                  </button>
                </div>
              </div>

              {/* Preferences form */}
              <div className="bg-white rounded-3xl p-5 shadow-md border border-purple-100 space-y-4">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                  <Palette size={18} className="text-purple-500" />
                  선호 무드 선택 (최대 3개)
                </h3>
                <div className="flex flex-wrap gap-2">
                  {MOOD_OPTIONS.map((option) => {
                    const active = preferredMoods.includes(option.id);
                    return (
                      <button
                        key={option.id}
                        onClick={() => toggleMood(option.id)}
                        className={`px-4 py-2 rounded-full border text-xs font-semibold transition-all ${
                          active
                            ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white border-transparent shadow'
                            : 'border-purple-200 text-purple-600 hover:border-purple-400'
                        }`}
                      >
                        #{option.label}
                      </button>
                    );
                  })}
                </div>
                <button
                  onClick={handlePreferencesSave}
                  disabled={prefsSaving}
                  className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-2xl hover:from-purple-600 hover:to-pink-600 transition-all shadow-lg disabled:opacity-50"
                >
                  {prefsSaving ? '저장 중...' : '선호 무드 저장'}
                </button>
              </div>
            </div>

            {/* Right Column */}
            <div className="flex flex-col h-full overflow-hidden">
              <div className="flex items-center justify-between mb-6 flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-pink-400 to-rose-400 rounded-2xl flex items-center justify-center shadow-sm">
                    <Heart size={24} className="text-white fill-white" />
                  </div>
                  <h2 className="text-2xl text-gray-800">관심 상품</h2>
                  <span className="px-4 py-2 bg-gradient-to-r from-pink-500 to-rose-500 text-white rounded-full text-sm shadow-sm">
                    {favorites.length}개
                  </span>
                </div>

                {favorites.length > 0 && (
                  <div className="relative">
                    <button
                      onClick={() => setShowSortMenu(!showSortMenu)}
                      className="px-6 py-3 bg-white border-2 border-blue-200 rounded-2xl hover:border-blue-400 transition-all shadow-sm flex items-center gap-2"
                    >
                      <ArrowUpDown size={18} className="text-blue-600" />
                      <span className="text-gray-700">{getSortLabel()}</span>
                    </button>

                    {showSortMenu && (
                      <div className="absolute right-0 top-full mt-2 bg-white rounded-2xl shadow-lg border-2 border-blue-100 overflow-hidden z-10 min-w-[160px]">
                        <button
                          onClick={() => {
                            setSortBy('recent');
                            setShowSortMenu(false);
                          }}
                          className={`w-full text-left px-5 py-3 hover:bg-blue-50 transition-colors ${
                            sortBy === 'recent' ? 'bg-blue-50 text-blue-600' : 'text-gray-700'
                          }`}
                        >
                          등록순
                        </button>
                        <button
                          onClick={() => {
                            setSortBy('price-low');
                            setShowSortMenu(false);
                          }}
                          className={`w-full text-left px-5 py-3 hover:bg-blue-50 transition-colors ${
                            sortBy === 'price-low' ? 'bg-blue-50 text-blue-600' : 'text-gray-700'
                          }`}
                        >
                          가격 낮은순
                        </button>
                        <button
                          onClick={() => {
                            setSortBy('price-high');
                            setShowSortMenu(false);
                          }}
                          className={`w-full text-left px-5 py-3 hover:bg-blue-50 transition-colors ${
                            sortBy === 'price-high' ? 'bg-blue-50 text-blue-600' : 'text-gray-700'
                          }`}
                        >
                          가격 높은순
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="flex-1 bg-white rounded-3xl shadow-xl border-2 border-pink-100 overflow-hidden flex flex-col min-h-0">
                <div className="h-full overflow-y-auto p-6">
                  {favorites.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center">
                      <div className="w-32 h-32 bg-gradient-to-br from-pink-100 to-rose-100 rounded-full flex items-center justify-center mb-6">
                        <Heart size={64} className="text-pink-300" />
                      </div>
                      <h3 className="text-2xl mb-3 text-gray-800">아직 관심 상품이 없어요</h3>
                      <p className="text-gray-500 mb-8 text-center">
                        챗봇에서 마음에 드는 상품을 찾아보세요!<br />
                        AI가 당신의 취향에 맞는 상품을 추천해드릴게요.
                      </p>
                      <button
                        onClick={() => onNavigate('chat')}
                        className="px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-1 flex items-center gap-2"
                      >
                        <span>챗봇으로 가기</span>
                        <span>💬</span>
                      </button>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-5">
                      {sortedProducts.map((item) => {
                        const product = item.product;
                        if (!product) return null;
                        const removing = favoriteRemovingId === item.id;
                        return (
                          <div key={item.id} className="group">
                            <div className="bg-white rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition-all border border-gray-200 hover:border-pink-200 transform hover:-translate-y-1">
                              <div className="relative aspect-square overflow-hidden">
                                {product.image_url ? (
                                  <img
                                    src={product.image_url}
                                    alt={product.product_name}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                  />
                                ) : (
                                  <div className="flex h-full items-center justify-center text-sm text-gray-400">
                                    이미지 없음
                                  </div>
                                )}
                                <div className="absolute top-3 left-3 px-3 py-1 bg-gradient-to-r from-pink-500 to-rose-500 text-white rounded-lg text-xs flex items-center gap-1">
                                  관심상품
                                </div>
                                <button
                                  onClick={() => handleFavoriteDelete(item.id)}
                                  disabled={removing}
                                  className="absolute top-3 right-3 p-1.5 bg-white/95 backdrop-blur rounded-lg hover:bg-pink-50 transition-all shadow-md group/btn"
                                  title="관심 상품 해제"
                                >
                                  {removing ? (
                                    <Loader2 size={16} className="animate-spin text-pink-500" />
                                  ) : (
                                    <Heart size={18} className="text-pink-500 fill-pink-500 group-hover/btn:scale-110 transition-transform" />
                                  )}
                                </button>
                              </div>
                              <div className="p-4">
                                <h4 className="mb-1 truncate text-gray-800">{product.product_name}</h4>
                                <p className="text-xs text-gray-500 mb-2">{product.brand_name || 'Brand'}</p>
                                <p className="text-blue-600 mb-3">{formatPrice(product.price)}</p>
                                {product.link_url && (
                                  <a
                                    href={product.link_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all text-sm"
                                  >
                                    <ExternalLink size={16} />
                                    <span>구매하러 가기</span>
                                  </a>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {toastMessage && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 rounded-full bg-black/80 px-6 py-3 text-sm text-white shadow-lg z-50">
          {toastMessage}
        </div>
      )}

      <DeleteAccountDialog
        open={showDeleteAccountPopup}
        onClose={() => setShowDeleteAccountPopup(false)}
        password={deletePassword}
        onPasswordChange={setDeletePassword}
        error={deleteError}
        onConfirm={confirmDeleteAccount}
        loading={isDeleting}
      />

      <SuccessDialog open={showSuccessPopup} onClose={handleSuccessConfirm} />
    </div>
  );
}

function InfoTile({
  label,
  value,
  icon,
  emoji,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  emoji: React.ReactNode;
}) {
  return (
    <div className="group flex flex-col">
      <div className="relative flex-1 mb-1.5 min-h-[100px]">
        <div className="w-full h-full bg-gradient-to-br from-purple-200 via-purple-300 to-purple-400 rounded-xl flex items-center justify-center shadow-md group-hover:shadow-lg transition-all text-3xl">
          {emoji}
        </div>
        <div className="absolute top-2 right-2 w-5 h-5 bg-white rounded-full shadow-md flex items-center justify-center">
          {icon}
        </div>
      </div>
      <div className="text-center">
        <p className="text-[10px] text-gray-500 mb-0.5">{label}</p>
        <p className="text-xs font-semibold text-gray-800 truncate">{value}</p>
      </div>
    </div>
  );
}

function DeleteAccountDialog({
  open,
  onClose,
  password,
  onPasswordChange,
  error,
  onConfirm,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  password: string;
  onPasswordChange: (value: string) => void;
  error: string;
  onConfirm: () => void;
  loading: boolean;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-3xl p-8 max-w-lg w-full shadow-2xl">
        <div className="text-center mb-6">
          <div className="w-20 h-20 bg-gradient-to-br from-red-400 to-rose-400 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
            <AlertCircle size={40} className="text-white" />
          </div>
          <h2 className="text-2xl mb-3 text-gray-800">정말 탈퇴하시겠어요?</h2>
          <p className="text-gray-600 mb-6">
            회원 탈퇴 시 모든 데이터가 삭제되며<br />
            복구할 수 없습니다.
          </p>
        </div>
        <div className="bg-red-50 rounded-2xl p-5 mb-6 border-2 border-red-100">
          <h3 className="text-sm mb-3 text-red-800 flex items-center gap-2">
            <AlertCircle size={18} />
            <span>삭제되는 정보</span>
          </h3>
          <ul className="text-sm text-gray-700 space-y-2">
            <li>계정 정보 및 프로필</li>
            <li>취향 분석 데이터</li>
            <li>관심 상품 목록</li>
            <li>채팅 히스토리</li>
          </ul>
        </div>
        <div className="mb-6">
          <label className="text-sm text-gray-700 mb-2 block">비밀번호 확인</label>
          <input
            type="password"
            value={password}
            onChange={(e) => onPasswordChange(e.target.value)}
            placeholder="비밀번호를 입력하세요"
            className="w-full rounded-2xl border border-gray-300 px-4 py-3 focus:border-red-400 focus:outline-none"
          />
          {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
        </div>
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-4 border-2 border-gray-300 rounded-2xl hover:bg-gray-50 transition-all text-gray-700"
          >
            취소
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className="flex-1 py-4 bg-gradient-to-r from-red-500 to-rose-500 text-white rounded-2xl hover:from-red-600 hover:to-rose-600 transition-all shadow-lg disabled:opacity-50"
          >
            {loading ? '처리 중...' : '탈퇴하기'}
          </button>
        </div>
      </div>
    </div>
  );
}

function SuccessDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
        <div className="text-center mb-6">
          <div className="w-20 h-20 bg-gradient-to-br from-green-400 to-emerald-400 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
            <span className="text-4xl">✓</span>
          </div>
          <h2 className="text-2xl mb-3 text-gray-800">회원 탈퇴 완료</h2>
          <p className="text-gray-600">
            회원 탈퇴가 완료되었습니다.<br />
            그동안 MOOD ON을 이용해주셔서 감사합니다.
          </p>
        </div>
        <button
          onClick={onClose}
          className="w-full py-4 bg-gradient-to-r from-blue-500 to-blue-400 text-white rounded-2xl hover:from-blue-600 hover:to-blue-500 transition-all shadow-lg"
        >
          확인
        </button>
      </div>
    </div>
  );
}

