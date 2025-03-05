// 오디오 파일 경로 설정
const hoverSound = new Audio('/static/sounds/hover1.mp3'); // 마우스 오버 시
const clickSound1 = new Audio('/static/sounds/click1.mp3'); // 버튼 클릭 시

// 오디오 미리 로드
hoverSound.preload = 'auto';
clickSound1.preload = 'auto';

// 사운드 활성화 상태
let soundEnabled = false;

document.addEventListener('DOMContentLoaded', () => {
    const audioToggleButton = document.getElementById('audio-toggle-button');

    // 버튼 클릭 이벤트 등록
    audioToggleButton.addEventListener('click', () => {
        soundEnabled = !soundEnabled; // 상태 토글

        if (soundEnabled) {
            audioToggleButton.textContent = '🔇 음소거'; // 텍스트 변경
        } else {
            audioToggleButton.textContent = '🔊 소리 허용'; // 텍스트 변경
        }
    });

    const buttons = document.querySelectorAll('button');

    buttons.forEach(button => {
        // 마우스 오버 시 사운드 재생
        button.addEventListener('mouseenter', () => {
            if (soundEnabled) {
                hoverSound.currentTime = 0;
                hoverSound.play();
            }
        });

        // 클릭 시 사운드 재생 및 페이지 이동
        button.addEventListener('mousedown', (event) => {
            event.preventDefault(); // 기본 동작 막기
            const href = button.getAttribute('href');

            console.log('버튼 클릭함', href);
            
            if (href === null) {
                return;
            }
            
            if (soundEnabled) {
                clickSound1.currentTime = 0;
                // 클릭 사운드 재생 후 페이지 이동
                clickSound1.play().then(() => {
                    // 사운드 재생이 시작되면 딜레이 후 페이지 이동
                    setTimeout(() => {
                        window.location.href = href;
                    }, 2000000); // 클릭 소리 길이에 따라 조정
                }).catch((error) => {
                    console.error('클릭 사운드 재생 실패:', error);
                    window.location.href = href;
                });
            } else {
                // 사운드가 비활성화된 경우 바로 페이지 이동
                console.log('사운드 비활성화 되어 있음');
                window.location.href = href;
            }
        });
    });
});