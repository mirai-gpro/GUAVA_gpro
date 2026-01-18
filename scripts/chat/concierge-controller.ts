// src/scripts/chat/concierge-controller.ts
import { CoreController } from './core-controller';
import { AudioManager } from './audio-manager';
import { GVRM } from '../../gvrm-format/gvrm';

declare const io: any;

export class ConciergeController extends CoreController {
  
  private guavaRenderer: GVRM | null = null;
  private analysisContext: AudioContext | null = null;
  private audioAnalyser: AnalyserNode | null = null;
  private analysisSource: MediaElementAudioSourceNode | null = null;
  private animationFrameId: number | null = null;

  constructor(container: HTMLElement, apiBase: string) {
    super(container, apiBase);
    this.audioManager = new AudioManager(8000);
    this.currentMode = 'concierge';
    this.init();
  }

  protected async init() {
    await super.init();
    
    const query = (sel: string) => this.container.querySelector(sel) as HTMLElement;
    this.els.avatarContainer = query('#avatar3DContainer'); 
    this.els.modeSwitch = query('#modeSwitch') as HTMLInputElement;
    
    if (this.els.avatarContainer) {
      try {
        this.guavaRenderer = new GVRM();
        
        const config = {
          templatePath: '/assets/avatar_web.ply',
          imagePath: '/assets/source.png'
        };
        
        await this.guavaRenderer.init(config);
        
        this.els.avatarContainer.classList.add('loaded');
        const fallback = document.getElementById('avatarFallback');
        if (fallback) fallback.style.display = 'none';
        console.log('[GVRM] ✅ Initialization successful');
        
      } catch (error) {
        console.error('[GVRM] Initialization error:', error);
        this.els.avatarContainer.classList.add('fallback');
      }
    }

    if (this.els.modeSwitch) {
      this.els.modeSwitch.addEventListener('change', () => {
        this.toggleMode();
      });
    }
  }

  protected async initializeSession() {
    try {
      if (this.sessionId) {
        try {
          await fetch(`${this.apiBase}/api/session/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId })
          });
        } catch (e) {}
      }

      const userId = this.getUserId();

      const res = await fetch(`${this.apiBase}/api/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_info: { user_id: userId },
          language: this.currentLanguage,
          mode: 'concierge'
        })
      });
      const data = await res.json();
      this.sessionId = data.session_id;

      const greetingText = data.initial_message || this.t('initialGreetingConcierge');
      this.addMessage('assistant', greetingText, null, true);
      
      const ackTexts = [
        this.t('ackConfirm'), this.t('ackSearch'), this.t('ackUnderstood'), 
        this.t('ackYes'), this.t('ttsIntro')
      ];
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];
      
      const ackPromises = ackTexts.map(async (text) => {
        try {
          const ackResponse = await fetch(`${this.apiBase}/api/tts/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              text: text, language_code: langConfig.tts, voice_name: langConfig.voice 
            })
          });
          const ackData = await ackResponse.json();
          if (ackData.success && ackData.audio) {
            this.preGeneratedAcks.set(text, ackData.audio);
          }
        } catch (_e) { }
      });

      await Promise.all([
        this.speakTextGCP(greetingText), 
        ...ackPromises
      ]);
      
      this.els.userInput.disabled = false;
      this.els.sendBtn.disabled = false;
      this.els.micBtn.disabled = false;
      this.els.speakerBtn.disabled = false;
      this.els.speakerBtn.classList.remove('disabled');
      this.els.reservationBtn.classList.remove('visible');

    } catch (e) {
      console.error('[Session] Initialization error:', e);
    }
  }

  protected initSocket() {
    // @ts-ignore
    this.socket = io(this.apiBase || window.location.origin);
    
    this.socket.on('connect', () => { });
    
    this.socket.on('transcript', (data: any) => {
      const { text, is_final } = data;
      if (this.isAISpeaking) return;
      if (is_final) {
        this.handleStreamingSTTComplete(text);
        this.currentAISpeech = "";
      } else {
        this.els.userInput.value = text;
      }
    });

    this.socket.on('error', (data: any) => {
      this.addMessage('system', `${this.t('sttError')} ${data.message}`);
      if (this.isRecording) this.stopStreamingSTT();
    });
  }

  protected async speakTextGCP(text: string, stopPrevious: boolean = true, autoRestartMic: boolean = false, skipAudio: boolean = false) {
    if (skipAudio || !this.isTTSEnabled || !text) return Promise.resolve();

    if (stopPrevious) {
      this.stopCurrentAudio();
    }
    
    this.setupAudioAnalysis();

    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.add('speaking');
    }
    
    await super.speakTextGCP(text, stopPrevious, autoRestartMic, skipAudio);
    
    this.stopAvatarAnimation();
  }

  private setupAudioAnalysis() {
    if (!this.guavaRenderer) return;

    if (!this.analysisContext) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      this.analysisContext = new AudioContextClass();
    }

    if (this.analysisContext.state === 'suspended') {
      this.analysisContext.resume().catch(e => console.log('AudioContext resume failed:', e));
    }

    if (!this.audioAnalyser) {
      this.audioAnalyser = this.analysisContext.createAnalyser();
      this.audioAnalyser.fftSize = 256;
    }

    if (!this.analysisSource && this.ttsPlayer) {
      try {
        this.analysisSource = this.analysisContext.createMediaElementSource(this.ttsPlayer);
        this.analysisSource.connect(this.audioAnalyser);
        this.audioAnalyser.connect(this.analysisContext.destination);
      } catch (e) {
        console.warn('MediaElementSource connection error:', e);
      }
    }

    this.startLipSyncLoop();
  }

  private startLipSyncLoop() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

    const update = () => {
      if (this.ttsPlayer.paused || this.ttsPlayer.ended) {
        this.guavaRenderer?.updateLipSync(0);
        
        if (this.ttsPlayer.ended) {
           this.animationFrameId = null;
           return; 
        }
      }

      if (this.audioAnalyser && this.guavaRenderer && !this.ttsPlayer.paused) {
        const dataArray = new Uint8Array(this.audioAnalyser.frequencyBinCount);
        this.audioAnalyser.getByteFrequencyData(dataArray);
        
        let sum = 0;
        const range = dataArray.length; 
        for (let i = 0; i < range; i++) {
          sum += dataArray[i];
        }
        const average = sum / range;
        
        const normalizedLevel = Math.min(1.0, (average / 255.0) * 2.5);

        this.guavaRenderer.updateLipSync(normalizedLevel);
      }
      
      this.animationFrameId = requestAnimationFrame(update);
    };

    this.animationFrameId = requestAnimationFrame(update);
  }

  private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.remove('speaking');
    }
    this.guavaRenderer?.updateLipSync(0);
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  protected updateUILanguage() {
    const initialMessage = this.els.chatArea.querySelector('.message.assistant[data-initial="true"] .message-text');
    const savedGreeting = initialMessage?.textContent;

    super.updateUILanguage();

    if (initialMessage && savedGreeting) {
      initialMessage.textContent = savedGreeting;
    }

    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
      pageTitle.innerHTML = `<img src="/pwa-152x152.png" alt="Logo" class="app-logo" /> ${this.t('pageTitleConcierge')}`;
    }
  }

  private toggleMode() {
    const isChecked = this.els.modeSwitch?.checked;
    if (!isChecked) {
      console.log('[ConciergeController] Switching to Chat mode...');
      window.location.href = '/';
    }
  }

  protected stopAllActivities() {
    super.stopAllActivities();
    this.stopAvatarAnimation();
  }

  private splitIntoSentences(text: string, language: string): string[] {
    let separator: RegExp;

    if (language === 'ja' || language === 'zh') {
      separator = /。/;
    } else {
      separator = /\.\s+/;
    }

    const sentences = text.split(separator).filter(s => s.trim().length > 0);

    return sentences.map((s, idx) => {
      if (idx < sentences.length - 1 || text.endsWith('。') || text.endsWith('. ')) {
        return language === 'ja' || language === 'zh' ? s + '。' : s + '. ';
      }
      return s;
    });
  }

  private async speakResponseInChunks(response: string, isTextInput: boolean = false) {
    if (isTextInput || !this.isTTSEnabled) {
      return this.speakTextGCP(response, true, false, isTextInput);
    }

    try {
      this.isAISpeaking = true;
      if (this.isRecording) {
        this.stopStreamingSTT();
      }
      
      this.setupAudioAnalysis();

      const sentences = this.splitIntoSentences(response, this.currentLanguage);

      if (sentences.length <= 1) {
        await this.speakTextGCP(response, true, false, isTextInput);
        this.isAISpeaking = false;
        return;
      }

      const firstSentence = sentences[0];
      const remainingSentences = sentences.slice(1).join('');
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      let firstSentenceAudioPromise: Promise<string | null> | null = null;
      let remainingAudioPromise: Promise<string | null> | null = null;

      if (this.isUserInteracted) {
        firstSentenceAudioPromise = (async () => {
          const cleanText = this.stripMarkdown(firstSentence);
          const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: cleanText,
              language_code: langConfig.tts,
              voice_name: langConfig.voice
            })
          });
          const result = await response.json();
          return result.success ? `data:audio/mp3;base64,${result.audio}` : null;
        })();

        if (remainingSentences.trim().length > 0) {
          remainingAudioPromise = (async () => {
            const cleanText = this.stripMarkdown(remainingSentences);
            const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                text: cleanText,
                language_code: langConfig.tts,
                voice_name: langConfig.voice
              })
            });
            const result = await response.json();
            return result.success ? `data:audio/mp3;base64,${result.audio}` : null;
          })();
        }

        if (firstSentenceAudioPromise) {
          const firstSentenceAudio = await firstSentenceAudioPromise;
          if (firstSentenceAudio) {
            const firstSentenceText = this.stripMarkdown(firstSentence);
            this.lastAISpeech = this.normalizeText(firstSentenceText);

            this.stopCurrentAudio();
            this.ttsPlayer.src = firstSentenceAudio;
            this.startLipSyncLoop();

            await new Promise<void>((resolve) => {
              this.ttsPlayer.onended = () => {
                this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                this.els.voiceStatus.className = 'voice-status stopped';
                resolve();
              };
              this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
              this.els.voiceStatus.className = 'voice-status speaking';
              this.ttsPlayer.play();
            });

            if (remainingAudioPromise) {
              const remainingAudio = await remainingAudioPromise;
              if (remainingAudio) {
                const remainingText = this.stripMarkdown(remainingSentences);
                this.lastAISpeech = this.normalizeText(remainingText);

                await new Promise(r => setTimeout(r, 300));

                this.stopCurrentAudio();
                this.ttsPlayer.src = remainingAudio;
                this.startLipSyncLoop();

                await new Promise<void>((resolve) => {
                  this.ttsPlayer.onended = () => {
                    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                    this.els.voiceStatus.className = 'voice-status stopped';
                    resolve();
                  };
                  this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
                  this.els.voiceStatus.className = 'voice-status speaking';
                  this.ttsPlayer.play();
                });
              }
            }
          }
        }
      }
      
      this.stopAvatarAnimation();
      this.isAISpeaking = false;
    } catch (error) {
      console.error('[TTS並列処理エラー]', error);
      this.isAISpeaking = false;
      await this.speakTextGCP(response, true, false, isTextInput);
    }
  }

  protected async handleStreamingSTTComplete(transcript: string) {
    this.stopStreamingSTT();
    
    if ('mediaSession' in navigator) {
      try { navigator.mediaSession.playbackState = 'playing'; } catch (e) {}
    }
    
    this.els.voiceStatus.innerHTML = this.t('voiceStatusComplete');
    this.els.voiceStatus.className = 'voice-status';

    const normTranscript = this.normalizeText(transcript);
    if (this.isSemanticEcho(normTranscript, this.lastAISpeech)) {
        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
        this.els.voiceStatus.className = 'voice-status stopped';
        this.lastAISpeech = '';
        return;
    }

    this.els.userInput.value = transcript;
    this.addMessage('user', transcript);
    
    const textLength = transcript.trim().replace(/\s+/g, '').length;
    if (textLength < 2) {
        const msg = this.t('shortMsgWarning');
        this.addMessage('assistant', msg);
        if (this.isTTSEnabled && this.isUserInteracted) {
          await this.speakTextGCP(msg, true);
        } else { 
          await new Promise(r => setTimeout(r, 2000)); 
        }
        this.els.userInput.value = '';
        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
        this.els.voiceStatus.className = 'voice-status stopped';
        return;
    }

    const ackText = this.t('ackYes');
    const preGeneratedAudio = this.preGeneratedAcks.get(ackText);
    
    let firstAckPromise: Promise<void> | null = null;
    if (preGeneratedAudio && this.isTTSEnabled && this.isUserInteracted) {
      firstAckPromise = new Promise<void>((resolve) => {
        this.setupAudioAnalysis();
        this.lastAISpeech = this.normalizeText(ackText);
        this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
        this.ttsPlayer.onended = () => resolve();
        this.ttsPlayer.play().catch(_e => resolve());
      });
    } else if (this.isTTSEnabled) { 
      firstAckPromise = this.speakTextGCP(ackText, false); 
    }
    
    this.addMessage('assistant', ackText);
    
    (async () => {
      try {
        if (firstAckPromise) await firstAckPromise;
        if (this.els.userInput.value.trim()) {
          this.isFromVoiceInput = true;
          this.sendMessage();
        }
      } catch (_error) {
        if (this.els.userInput.value.trim()) {
          this.isFromVoiceInput = true;
          this.sendMessage();
        }
      }
    })();
    
    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
    this.els.voiceStatus.className = 'voice-status stopped';
  }

  protected async sendMessage() {
    let firstAckPromise: Promise<void> | null = null; 
    this.unlockAudioParams();
    const message = this.els.userInput.value.trim();
    if (!message || this.isProcessing) return;
    
    const currentSessionId = this.sessionId;
    const isTextInput = !this.isFromVoiceInput;
    
    this.isProcessing = true; 
    this.els.sendBtn.disabled = true;
    this.els.micBtn.disabled = true; 
    this.els.userInput.disabled = true;

    if (!this.isFromVoiceInput) {
      this.addMessage('user', message);
      const textLength = message.trim().replace(/\s+/g, '').length;
      if (textLength < 2) {
           const msg = this.t('shortMsgWarning');
           this.addMessage('assistant', msg);
           if (this.isTTSEnabled && this.isUserInteracted) await this.speakTextGCP(msg, true);
           this.resetInputState();
           return;
      }
      
      this.els.userInput.value = '';
      
      const ackText = this.t('ackYes');
      this.currentAISpeech = ackText;
      this.addMessage('assistant', ackText);
      
      if (this.isTTSEnabled && !isTextInput) {
        try {
          const preGeneratedAudio = this.preGeneratedAcks.get(ackText);
          if (preGeneratedAudio && this.isUserInteracted) {
            firstAckPromise = new Promise<void>((resolve) => {
              this.setupAudioAnalysis();
              this.lastAISpeech = this.normalizeText(ackText);
              this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
              this.ttsPlayer.onended = () => resolve();
              this.ttsPlayer.play().catch(_e => resolve());
            });
          } else { 
            firstAckPromise = this.speakTextGCP(ackText, false); 
          }
        } catch (_e) {}
      }   
      if (firstAckPromise) await firstAckPromise;
    }

    this.isFromVoiceInput = false;
    
    if (this.waitOverlayTimer) clearTimeout(this.waitOverlayTimer);
    let responseReceived = false;
    
    this.waitOverlayTimer = window.setTimeout(() => { 
      if (!responseReceived) {
        this.showWaitOverlay(); 
      }
    }, 6500);

    try {
      const response = await fetch(`${this.apiBase}/api/chat`, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ 
          session_id: currentSessionId, 
          message: message, 
          stage: this.currentStage, 
          language: this.currentLanguage,
          mode: this.currentMode
        }) 
      });
      const data = await response.json();
      responseReceived = true;
      
      if (this.sessionId !== currentSessionId) return;
      
      if (this.waitOverlayTimer) {
        clearTimeout(this.waitOverlayTimer);
        this.waitOverlayTimer = null;
      }
      this.hideWaitOverlay();
      this.currentAISpeech = data.response;
      this.addMessage('assistant', data.response, data.summary);
      
      if (!isTextInput && this.isTTSEnabled) {
        this.stopCurrentAudio();
      }
      
      if (data.shops && data.shops.length > 0) {
        this.currentShops = data.shops;
        this.els.reservationBtn.classList.add('visible');
        this.els.userInput.value = '';
        document.dispatchEvent(new CustomEvent('displayShops', { 
          detail: { shops: data.shops, language: this.currentLanguage } 
        }));
        
        const section = document.getElementById('shopListSection');
        if (section) section.classList.add('has-shops');
        if (window.innerWidth < 1024) {
          setTimeout(() => {
            const shopSection = document.getElementById('shopListSection');
            if (shopSection) shopSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
           }, 300);
        }
        
        (async () => {
          try {
            this.isAISpeaking = true;
            if (this.isRecording) { this.stopStreamingSTT(); }

            await this.speakTextGCP(this.t('ttsIntro'), true, false, isTextInput);
            
            const lines = data.response.split('\n\n');
            let introText = ""; 
            let shopLines = lines;
            if (lines[0].includes('ご希望に合うお店') && lines[0].includes('ご紹介します')) { 
              introText = lines[0]; 
              shopLines = lines.slice(1); 
            }
            
            let introPart2Promise: Promise<void> | null = null;
            if (introText && this.isTTSEnabled && this.isUserInteracted && !isTextInput) {
                const preGeneratedIntro = this.preGeneratedAcks.get(introText);
              if (preGeneratedIntro) {
                introPart2Promise = new Promise<void>((resolve) => {
                  this.setupAudioAnalysis();
                  this.lastAISpeech = this.normalizeText(introText);
                  this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedIntro}`;
                  this.ttsPlayer.onended = () => resolve();
                  this.ttsPlayer.play();
                });
              } else { 
                introPart2Promise = this.speakTextGCP(introText, false, false, isTextInput); 
              }
            }

            let firstShopAudioPromise: Promise<string | null> | null = null;
            let remainingAudioPromise: Promise<string | null> | null = null;
            const shopLangConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];
            
            if (shopLines.length > 0 && this.isTTSEnabled && this.isUserInteracted && !isTextInput) {
              const firstShop = shopLines[0];
              const restShops = shopLines.slice(1).join('\n\n');              
              firstShopAudioPromise = (async () => {
                const cleanText = this.stripMarkdown(firstShop);
                const response = await fetch(`${this.apiBase}/api/tts/synthesize`, { 
                  method: 'POST', 
                  headers: { 'Content-Type': 'application/json' }, 
                  body: JSON.stringify({ 
                    text: cleanText, language_code: shopLangConfig.tts, voice_name: shopLangConfig.voice 
                  }) 
                });
                const result = await response.json();
                return result.success ? `data:audio/mp3;base64,${result.audio}` : null;
              })();
              
              if (restShops) {
                remainingAudioPromise = (async () => {
                  const cleanText = this.stripMarkdown(restShops);
                  const response = await fetch(`${this.apiBase}/api/tts/synthesize`, { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ 
                      text: cleanText, language_code: shopLangConfig.tts, voice_name: shopLangConfig.voice 
                    }) 
                  });
                  const result = await response.json();
                  return result.success ? `data:audio/mp3;base64,${result.audio}` : null;
                })();
              }
            }

            if (introPart2Promise) await introPart2Promise;
            
            if (firstShopAudioPromise) {
              const firstShopAudio = await firstShopAudioPromise;
              if (firstShopAudio) {
                const firstShopText = this.stripMarkdown(shopLines[0]);
                this.lastAISpeech = this.normalizeText(firstShopText);
                
                if (!isTextInput && this.isTTSEnabled) {
                  this.stopCurrentAudio();
                }
                
                this.ttsPlayer.src = firstShopAudio;     
                this.setupAudioAnalysis();           
                await new Promise<void>((resolve) => { 
                  this.ttsPlayer.onended = () => { 
                    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped'); 
                    this.els.voiceStatus.className = 'voice-status stopped'; 
                    resolve(); 
                  }; 
                  this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking'); 
                  this.els.voiceStatus.className = 'voice-status speaking'; 
                  this.ttsPlayer.play(); 
                });
                
                if (remainingAudioPromise) {
                  const remainingAudio = await remainingAudioPromise;
                  if (remainingAudio) {
                    const restShopsText = this.stripMarkdown(shopLines.slice(1).join('\n\n'));
                    this.lastAISpeech = this.normalizeText(restShopsText);
                    await new Promise(r => setTimeout(r, 500));
                    
                    if (!isTextInput && this.isTTSEnabled) {
                      this.stopCurrentAudio();
                    }
                    
                    this.ttsPlayer.src = remainingAudio;    
                    this.setupAudioAnalysis();                
                    await new Promise<void>((resolve) => { 
                      this.ttsPlayer.onended = () => { 
                        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped'); 
                        this.els.voiceStatus.className = 'voice-status stopped'; 
                        resolve(); 
                      }; 
                      this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking'); 
                      this.els.voiceStatus.className = 'voice-status speaking'; 
                      this.ttsPlayer.play(); 
                    });
                  }
                }
              }
            }
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
          } catch (_e) { 
            this.isAISpeaking = false; 
            this.stopAvatarAnimation();
          }
        })();
      } else {
        if (data.response) {
          const extractedShops = this.extractShopsFromResponse(data.response);
          if (extractedShops.length > 0) {
            this.currentShops = extractedShops;
            this.els.reservationBtn.classList.add('visible');
            document.dispatchEvent(new CustomEvent('displayShops', {
              detail: { shops: extractedShops, language: this.currentLanguage }
            }));
            const section = document.getElementById('shopListSection');
            if (section) section.classList.add('has-shops');
            this.speakResponseInChunks(data.response, isTextInput);
          } else {
            this.speakResponseInChunks(data.response, isTextInput);
          }
        }
      }
    } catch (error) { 
      console.error('送信エラー:', error);
      this.hideWaitOverlay(); 
      this.showError('メッセージの送信に失敗しました。'); 
    } finally { 
      this.resetInputState();
      this.els.userInput.blur();
    }
  }
}