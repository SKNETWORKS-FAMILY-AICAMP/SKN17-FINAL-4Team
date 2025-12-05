# chat/urls.py

from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    # 세션 목록 조회 / 생성 (REQ-CHT-002, 005)
    # GET  /api/chat/sessions/
    # POST /api/chat/sessions/
    path(
        "sessions/",
        views.ChatSessionListCreateView.as_view(),
        name="chat-session-list-create",
    ),

    # 세션 상세 조회 / 삭제 (REQ-CHT-003, 004, 007)
    # GET    /api/chat/sessions/<id>/
    # DELETE /api/chat/sessions/<id>/
    path(
        "sessions/<int:pk>/",
        views.ChatSessionDetailView.as_view(),
        name="chat-session-detail",
    ),

    # 세션 상태 조회 / 수정 (REQ-CHT-003, 004)
    # GET   /api/chat/sessions/<session_id>/state/
    # PATCH /api/chat/sessions/<session_id>/state/
    path(
        "sessions/<int:session_id>/state/",
        views.SessionStateView.as_view(),
        name="chat-session-state",
    ),

    # 세션 내 메시지 조회 / 생성 (REQ-CHT-001, 005)
    # GET  /api/chat/sessions/<session_id>/messages/
    # POST /api/chat/sessions/<session_id>/messages/
    path(
        "sessions/<int:session_id>/messages/",
        views.ChatMessageListCreateView.as_view(),
        name="chat-message-list-create",
    ),

    # 추천 응답에 대한 만족도 저장 (REQ-CHT-001 일부)
    # POST /api/chat/messages/<message_id>/satisfaction/
    path(
        "messages/<int:message_id>/satisfaction/",
        views.ChatMessageSatisfactionView.as_view(),
        name="chat-message-satisfaction",
    ),

    # 세션 리셋 (상태 + 메시지 삭제, 모델 서버 세션도 리셋)
    # POST /api/chat/sessions/<session_id>/reset/
    path(
        "sessions/<int:session_id>/reset/",
        views.ResetSessionView.as_view(),
        name="chat-session-reset",
    ),
]
