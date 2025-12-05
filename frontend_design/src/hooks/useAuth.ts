import { useState, useEffect, useCallback } from 'react';

import { api, initCsrfToken } from '../lib/api';
import type { UserProfile } from '../types/user';

interface LoginPayload {
  email: string;
  password: string;
}

export function useAuth() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [initializing, setInitializing] = useState(true);

  const fetchSession = useCallback(async () => {
    // TODO(auth-session): 리프레시 후에도 세션 쿠키 기반으로 자동 로그인 되는지 통합 테스트 케이스 작성 예정.
    try {
      await initCsrfToken();
      const res = await api.get('/accounts/session/');
      if (res.data?.is_authenticated) {
        setUser(res.data.user);
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error('세션 정보를 불러오지 못했습니다.', error);
    } finally {
      setInitializing(false);
    }
  }, []);

  useEffect(() => {
    fetchSession();
  }, [fetchSession]);

  const login = useCallback(async ({ email, password }: LoginPayload) => {
    await initCsrfToken();
    const res = await api.post('/accounts/login/', { email, password });
    setUser(res.data?.user ?? null);
    return res.data;
  }, []);

  const logout = useCallback(async () => {
    await initCsrfToken();
    await api.post('/accounts/logout/');
    setUser(null);
  }, []);

  const deleteAccount = useCallback(async (payload: { password: string }) => {
    await initCsrfToken();
    await api.post('/accounts/delete/', payload);
    setUser(null);
  }, []);

  const updateProfileLocally = useCallback((next: UserProfile | null) => {
    setUser(next);
  }, []);

  return {
    user,
    isAuthenticated: Boolean(user),
    initializing,
    login,
    logout,
    deleteAccount,
    refreshSession: fetchSession,
    setUser: updateProfileLocally,
  };
}

