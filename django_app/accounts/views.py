# accounts/views.py

from datetime import timedelta
import random
import string

from django.contrib.auth import authenticate, get_user_model, login, logout
from django.utils import timezone
from django.middleware.csrf import get_token
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import EmailVerification
from .serializers import (
    UserSerializer,
    UserProfileSerializer,
    DeleteAccountSerializer,
    LoginSerializer,
    PasswordChangeSerializer,
    PasswordResetCompleteSerializer,
    PasswordResetRequestSerializer,
    PasswordResetVerifySerializer,
    RegisterEmailSerializer,
    RegisterSerializer,
    RegisterVerifySerializer,
)
from .utils import send_verification_email

User = get_user_model()

MAX_FAILED_LOGIN = 5  # 5회 이상 실패 시 10분 잠금


def generate_code(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


# ---------- 회원가입: 이메일 인증 코드 발송 ----------


class SendSignupCodeView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterEmailSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data["email"]

        code = generate_code()
        try:
            ev = EmailVerification.create_new(email=email, purpose="signup", code=code)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)

        send_verification_email(email=email, code=code, purpose="signup")
        return Response({"detail": "인증 코드가 발송되었습니다."}, status=status.HTTP_200_OK)


# ---------- 회원가입: 코드 검증 ----------


class VerifySignupCodeView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        ev: EmailVerification = serializer.validated_data["verification"]
        ev.mark_verified()
        return Response({"detail": "이메일 인증이 완료되었습니다."}, status=status.HTTP_200_OK)


# ---------- 회원가입: 실제 계정 생성 ----------


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        user_data = UserSerializer(user).data
        return Response(
            {
                "detail": "회원가입이 완료되었습니다.",
                "user": user_data,
            },
            status=status.HTTP_201_CREATED,
        )


# ---------- 로그인 / 로그아웃 ----------


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]

        user = User.objects.filter(email=email).first()
        if user and user.is_locked:
            return Response(
                {"detail": "로그인 시도 제한. 잠시 후 다시 시도해 주세요."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user = authenticate(request, email=email, password=password)
        if user is None:
            # 실패 처리
            user_for_lock = User.objects.filter(email=email).first()
            if user_for_lock:
                user_for_lock.register_failed_login(
                    max_failed=MAX_FAILED_LOGIN, lock_minutes=10
                )
            return Response(
                {"detail": "이메일 또는 비밀번호가 일치하지 않습니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 성공 → 잠금 초기화
        user.reset_login_lock()
        login(request, user)
        user_data = UserSerializer(user).data
        return Response(
            {
                "detail": "로그인 성공",
                "user": user_data,
            },
            status=status.HTTP_200_OK,
        )


class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        logout(request)
        return Response({"detail": "로그아웃 되었습니다."}, status=status.HTTP_200_OK)


# ---------- 비밀번호 재설정 (비로그인) ----------


class PasswordResetRequestView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"]
        code = generate_code()
        try:
            EmailVerification.create_new(email=email, purpose="reset", code=code)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)

        send_verification_email(email=email, code=code, purpose="reset")
        return Response({"detail": "비밀번호 재설정용 인증 코드가 발송되었습니다."}, status=status.HTTP_200_OK)


class PasswordResetVerifyView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = PasswordResetVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        ev: EmailVerification = serializer.validated_data["verification"]
        ev.mark_verified()
        return Response({"detail": "이메일 인증이 완료되었습니다."}, status=status.HTTP_200_OK)


class PasswordResetCompleteView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = PasswordResetCompleteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]

        user = User.objects.get(email=email)
        user.set_password(password)
        user.save(update_fields=["password"])

        return Response({"detail": "비밀번호가 재설정되었습니다."}, status=status.HTTP_200_OK)


# ---------- 비밀번호 변경 (로그인 상태) ----------


class PasswordChangeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = PasswordChangeSerializer(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)

        user = request.user
        password = serializer.validated_data["password"]
        user.set_password(password)
        user.save(update_fields=["password"])

        return Response({"detail": "비밀번호가 변경되었습니다."}, status=status.HTTP_200_OK)


# ---------- 회원 탈퇴 ----------


class DeleteAccountView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = DeleteAccountSerializer(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)

        user = request.user
        user.delete()
        logout(request)
        return Response({"detail": "회원 탈퇴가 완료되었습니다."}, status=status.HTTP_200_OK)


class CSRFTokenView(APIView):
    """
    Frontend가 CSRF 토큰을 가져갈 수 있도록 지원 (REQ-USER-001 기반).
    """

    permission_classes = [permissions.AllowAny]

    def get(self, request):
        token = get_token(request)
        return Response({"csrfToken": token})


class SessionStatusView(APIView):
    """
    현재 로그인 상태 및 사용자 정보 확인용.
    """

    permission_classes = [permissions.AllowAny]

    def get(self, request):
        if request.user.is_authenticated:
            data = UserSerializer(request.user).data
            return Response({"is_authenticated": True, "user": data})
        return Response({"is_authenticated": False, "user": None})


class UserProfileView(APIView):
    """
    로그인한 사용자의 프로필 정보 조회/수정.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request):
        serializer = UserProfileSerializer(
            request.user,
            data=request.data,
            partial=True,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
