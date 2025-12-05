import React from 'react';
import { Header } from './Header';
import { ArrowLeft } from 'lucide-react';
import type { Page } from '../types/navigation';

interface ChatGuidelinesPageProps {
  onNavigate: (page: Page) => void;
  isLoggedIn: boolean;
  onLogout?: () => void;
}

export function ChatGuidelinesPage({ onNavigate, isLoggedIn, onLogout }: ChatGuidelinesPageProps) {
  return (
    <div className="min-h-screen bg-white">
      <Header onNavigate={onNavigate} isLoggedIn={isLoggedIn} onLogout={onLogout} />
      
      <main className="pt-20 pb-16">
        <div className="max-w-4xl mx-auto px-6">
          <button 
            onClick={() => onNavigate('chat')}
            className="flex items-center gap-2 mb-8 hover:opacity-70 transition-opacity"
          >
            <ArrowLeft size={20} />
            <span>챗봇으로 돌아가기</span>
          </button>

          <h1 className="text-4xl mb-8">MOOD ON AI 챗봇 이용 주의사항</h1>

          <div className="space-y-8">
            <section>
              <h2 className="text-2xl mb-4">1. 서비스 목적 및 한계</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• MOOD ON AI 챗봇은 인테리어 상품 추천 및 정보 제공을 위한 보조 도구입니다.</p>
                <p>• AI가 제공하는 추천은 참고용이며, 최종 구매 결정은 사용자의 책임입니다.</p>
                <p>• 실시간 재고, 가격, 배송 정보는 실제와 다를 수 있으니 구매 전 반드시 확인하세요.</p>
                <p>• 챗봇은 일반적인 인테리어 조언을 제공하며, 전문 인테리어 디자이너의 상담을 대체할 수 없습니다.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">2. 개인정보 보호</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• MOOD ON은 사용자의 개인정보를 보호하기 위해 최선을 다하고 있습니다.</p>
                <p>• 채팅 내용에 주민등록번호, 계좌번호, 비밀번호, 신용카드 정보 등의 민감한 개인정보를 입력하지 마세요.</p>
                <p>• 정확한 주소나 전화번호 등의 개인 식별 정보 입력을 자제해주세요.</p>
                <p>• AI 챗봇과의 대화 내용은 서비스 개선을 위해 저장될 수 있습니다.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">3. AI 응답의 정확성</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• AI는 때때로 부정확하거나 오래된 정보를 제공할 수 있습니다.</p>
                <p>• 상품 정보, 가격, 재고 등은 실시간으로 변경될 수 있습니다.</p>
                <p>• 중요한 결정을 내리기 전에는 반드시 공식 정보를 확인하세요.</p>
                <p>• AI가 이해하기 어려운 질문의 경우 고객센터를 이용해주세요.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">4. 사용 제한사항</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• 질문은 한국어로만 입력 가능합니다.</p>
                <p>• 질문은 최대 200자 이내로 작성해주세요.</p>
                <p>• 이미지는 한 번에 1장만 업로드 가능하며, 최대 크기는 10MB입니다.</p>
                <p>• 과도한 요청이나 스팸성 질문은 제한될 수 있습니다.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">5. 금지된 행위</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• 불법적이거나 유해한 콘텐츠를 요청하는 행위</p>
                <p>• 타인의 권리를 침해하는 내용의 질문</p>
                <p>• 시스템의 취약점을 악용하려는 시도</p>
                <p>• 욕설, 비방, 차별적 표현 사용</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">6. 이미지 업로드 주의사항</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• 업로드하는 이미지에 개인정보가 포함되지 않도록 주의하세요.</p>
                <p>• 저작권이 있는 이미지를 무단으로 사용하지 마세요.</p>
                <p>• 지원 형식: JPG, PNG, GIF (최대 10MB)</p>
                <p>• 부적절한 이미지 업로드 시 계정이 제한될 수 있습니다.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">7. 서비스 이용 책임</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• AI 챗봇은 인테리어 상품 추천을 위한 보조 도구입니다.</p>
                <p>• 최종 구매 결정은 사용자의 책임이며, 회사는 이에 대한 법적 책임을 지지 않습니다.</p>
                <p>• 서비스는 예고 없이 변경되거나 중단될 수 있습니다.</p>
                <p>• 문제가 발생할 경우 고객센터로 문의해주세요.</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl mb-4">8. 채팅 히스토리</h2>
              <div className="bg-gray-50 p-6 rounded-2xl space-y-3">
                <p>• 채팅 히스토리는 사용자 계정에 저장됩니다.</p>
                <p>• 언제든지 과거 대화 내용을 확인할 수 있습니다.</p>
                <p>• 필요시 채팅 히스토리를 삭제할 수 있습니다.</p>
                <p>• 계정 탈퇴 시 모든 채팅 히스토리가 삭제됩니다.</p>
              </div>
            </section>

            <div className="bg-blue-50 p-6 rounded-2xl mt-8">
              <p className="text-center">
                위 내용을 숙지하고 올바르게 서비스를 이용해주세요.<br />
                더 나은 서비스를 제공하기 위해 항상 노력하겠습니다.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}