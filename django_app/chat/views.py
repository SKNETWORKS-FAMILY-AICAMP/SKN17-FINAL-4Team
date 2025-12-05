# chat/views.py

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import generics, permissions, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, ChatSession, SessionState
from .serializers import (
    ChatMessageCreateSerializer,
    ChatMessageSerializer,
    ChatSessionCreateSerializer,
    ChatSessionDetailSerializer,
    ChatSessionSerializer,
    SatisfactionSerializer,
    SessionStateSerializer,
)
from .services import (
    call_model_server_image,
    call_model_server_text,
    call_model_server_reset,
)


class ChatSessionListCreateView(generics.ListCreateAPIView):
    """
    GET: 히스토리 목록 조회 (REQ-CHT-005)
    POST: 새 채팅 생성 (REQ-CHT-002)
    """

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ChatSessionSerializer

    def get_queryset(self):
        return ChatSession.objects.filter(
            user=self.request.user, is_deleted=False
        ).order_by("-created_at")

    def get_serializer_class(self):
        if self.request.method == "POST":
            return ChatSessionCreateSerializer
        return ChatSessionSerializer


class ChatSessionDetailView(generics.RetrieveDestroyAPIView):
    """
    GET: 세션 상세 + 메시지 + 상태 조회 (REQ-CHT-003, 004)
    DELETE: 세션 삭제 (is_deleted 플래그) (REQ-CHT-007)
    """

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ChatSessionDetailSerializer

    def get_queryset(self):
        return ChatSession.objects.filter(
            user=self.request.user, is_deleted=False
        )

    def perform_destroy(self, instance: ChatSession):
        instance.is_deleted = True
        instance.save(update_fields=["is_deleted"])


class SessionStateView(APIView):
    """
    세션 상태 조회/수정 (REQ-CHT-003, 004)
    GET /api/chat/sessions/<session_id>/state/
    PATCH /api/chat/sessions/<session_id>/state/
    """

    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, session_id: int) -> SessionState:
        session = get_object_or_404(
            ChatSession,
            id=session_id,
            user=self.request.user,
            is_deleted=False,
        )
        state, _ = SessionState.objects.get_or_create(session=session)
        return state

    def get(self, request, session_id: int):
        state = self.get_object(session_id)
        serializer = SessionStateSerializer(state)
        return Response(serializer.data)

    def patch(self, request, session_id: int):
        state = self.get_object(session_id)
        serializer = SessionStateSerializer(state, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class ChatMessageListCreateView(APIView):
    """
    세션 내 메시지 목록 조회 및 생성.

    GET /api/chat/sessions/<session_id>/messages/
    POST /api/chat/sessions/<session_id>/messages/
    """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get_session_and_state(self, session_id: int) -> tuple[ChatSession, SessionState]:
        session = get_object_or_404(
            ChatSession,
            id=session_id,
            user=self.request.user,
            is_deleted=False,
        )
        state, _ = SessionState.objects.get_or_create(session=session)
        return session, state

    def get(self, request, session_id: int):
        """
        히스토리 탭에서 세션별 메시지 조회 (REQ-CHT-005)
        """
        session, state = self.get_session_and_state(session_id)
        messages = session.messages.order_by("created_at")
        serializer = ChatMessageSerializer(
            messages, many=True, context={"request": request}
        )
        state_serializer = SessionStateSerializer(state)
        return Response(
            {
                "session": ChatSessionSerializer(session).data,
                "messages": serializer.data,
                "session_state": state_serializer.data,
            }
        )

    def post(self, request, session_id: int):
        """
        채팅창에서 메시지 전송 (REQ-CHT-001)
        - 텍스트 또는 이미지(또는 둘 다)를 받아 model_server에 전달
        """
        session, state = self.get_session_and_state(session_id)

        # 이미지 용량 체크 (10MB 제한 예시)
        image_file = request.FILES.get("image")
        if image_file and image_file.size > 10 * 1024 * 1024:
            return Response(
                {"detail": "이미지 용량은 10MB 이하여야 합니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        has_image = bool(image_file)
        serializer = ChatMessageCreateSerializer(
            data=request.data,
            context={"request": request, "has_image": has_image},
        )
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        text = validated_data.get("text", "")
        image = validated_data.get("image")
        request_more = validated_data.get("request_more", False)

        # 사용자 메시지 저장
        user_msg = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.ROLE_USER,
            text=text,
            image=image,
        )

        # 세션 상태 스냅샷 (지금은 model_server 에 직접 전달하진 않지만 유지)
        state_payload = SessionStateSerializer(state).data

        # model_server 호출
        try:
            if image:
                # ImageField 저장 후 path 사용
                user_msg.refresh_from_db()  # image 저장 경로 보장
                result = call_model_server_image(
                    session_id=session.id,
                    image_path=user_msg.image.path,
                    state_payload=state_payload,
                )
            else:
                result = call_model_server_text(
                    session_id=session.id,
                    user_text=text,
                    state_payload=state_payload,
                    request_more=request_more,
                )
        except Exception as e:
            # 모델 서버 에러 시 간단한 안내 메시지 반환
            assistant_msg = ChatMessage.objects.create(
                session=session,
                role=ChatMessage.ROLE_ASSISTANT,
                text="죄송합니다. 현재 추천 서버에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
            res_serializer = ChatMessageSerializer(
                assistant_msg, context={"request": request}
            )
            return Response(
                {
                    "assistant_message": res_serializer.data,
                    "error": str(e),
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # model_server 응답 파싱
        assistant_text = result.get("assistant_text", "")
        recommended_products = result.get("recommended_products", [])
        updated_state = result.get("updated_session_state")

        # assistant 메시지 저장
        assistant_msg = ChatMessage.objects.create(
        session=session,
        role=ChatMessage.ROLE_ASSISTANT,
        text=assistant_text,
        # 빈 경우에는 None 말고 [] 를 저장하도록 변경
        recommended_products=recommended_products or [],
        )

        # 세션 상태 업데이트 (현재는 빈 dict 이므로 실질적 변경은 없음)
        if isinstance(updated_state, dict):
            for key, value in updated_state.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.save()

        session.updated_at = timezone.now()
        session.save(update_fields=["updated_at"])

        res_serializer = ChatMessageSerializer(
            assistant_msg, context={"request": request}
        )
        return Response(
            {
                "assistant_message": res_serializer.data,
                "session_state": SessionStateSerializer(state).data,
            },
            status=status.HTTP_200_OK,
        )


class ChatMessageSatisfactionView(APIView):
    """
    추천 상품에 대한 만족도 표시 (REQ-CHT-001 일부)
    POST /api/chat/messages/<id>/satisfaction/
    body: { "score": 1~5 }
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, message_id: int):
        msg = get_object_or_404(
            ChatMessage,
            id=message_id,
            session__user=request.user,
            session__is_deleted=False,
            role=ChatMessage.ROLE_ASSISTANT,
        )
        serializer = SatisfactionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        score = serializer.validated_data["score"]
        msg.satisfaction = score
        msg.save(update_fields=["satisfaction"])
        return Response({"detail": "만족도가 저장되었습니다."}, status=status.HTTP_200_OK)


class ResetSessionView(APIView):
    """
    세션 전체 리셋 (상태 + 메시지 삭제, 모델 서버 세션도 리셋).

    POST /api/chat/sessions/<session_id>/reset/
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, session_id: int):
        session = get_object_or_404(
            ChatSession,
            id=session_id,
            user=request.user,
            is_deleted=False,
        )

        # 1) FastAPI 모델 서버 세션 리셋 시도 (실패해도 그냥 넘어감)
        try:
            call_model_server_reset(session.id)
        except Exception:
            pass

        # 2) Django 쪽 상태/메시지 정리
        SessionState.objects.filter(session=session).delete()
        session.messages.all().delete()

        session.updated_at = timezone.now()
        session.save(update_fields=["updated_at"])

        return Response(
            {"detail": "세션이 초기화되었습니다."},
            status=status.HTTP_200_OK,
        )
