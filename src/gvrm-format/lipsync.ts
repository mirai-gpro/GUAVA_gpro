export class LipSyncAnalyzer {
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private dataArray: Uint8Array | null = null;
    private source: MediaElementAudioSourceNode | null = null;

    // 既存の <audio> 要素に接続する
    public connect(audioElement: HTMLAudioElement) {
        try {
            // AudioContextの作成（シングルトン推奨）
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            }

            // 既に接続されている場合は何もしない（多重接続防止）
            if (this.source) return;

            // 分析ノードの作成
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256; // サイズは小さめで十分

            // <audio> からソースを作成
            // ※注意: CORS制限のある外部音声URLの場合、audioElement.crossOrigin = "anonymous" が必要です
            this.source = this.audioContext.createMediaElementSource(audioElement);

            // ソース -> 分析機 -> スピーカー（出力）へ繋ぐ
            // これをしないと音が聞こえなくなります
            this.source.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);

            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            console.log("[LipSync] Connected to TTS Audio Element.");
        } catch (e) {
            console.error("[LipSync] Connection failed:", e);
        }
    }

    // 現在の音量レベル (0.0 ~ 1.0) を取得
    public getLevel(): number {
        if (!this.analyser || !this.dataArray) return 0;

        // 周波数データを取得
        this.analyser.getByteFrequencyData(this.dataArray);

        // 平均音量を計算 (人の声の帯域を中心に簡易計算)
        let sum = 0;
        const length = this.dataArray.length;
        for (let i = 0; i < length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / length;

        // 正規化と感度調整
        // 0~255 を 0.0~1.0 に。TTSはクリアなので感度は適度でOK
        const level = average / 255.0;
        
        // 小さすぎる音はカット（ノイズゲート）しつつ、少し増幅
        return level > 0.02 ? Math.min(1.0, level * 2.5) : 0;
    }
}
