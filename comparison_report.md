# TnsrLang vs. PyTorch: Analys och Jämförelse

Det stämmer! Med `Sequential`, `AutoGrad`, och inbyggt stöd för CNN/RNN/LSTM har **TnsrLang** blivit ett genuint (om än minimalistiskt) Deep Learning-ramverk som arkitektoniskt påminner väldigt mycket om PyTorch. 

Här är en rankning och jämförelse av de två språken, samt vilka flaskhalsar som just nu begränsar TnsrLangs prestanda.

---

## 1. Arkitektur och Funktioner

| Kategori | PyTorch | TnsrLang | Analys |
| :--- | :--- | :--- | :--- |
| **Graf-byggande** | Dynamic (Define-by-run) | Dynamic (Define-by-run) | **Lika!** Båda använder en "tejp" (tape) där varje operation dynamiskt registrerar hur dess bakåtpass ska räknas ut. TnsrLangs `children` och `backward_op` är en direkt motsvarighet till PyTorchs `autograd::Node`. |
| **API & Syntax** | Objektorienterat (`nn.Module`) | Funktionellt/Keras-stil (`Sequential`) | TnsrLang bygger på en array-baserad config-matris. Det är extremt lättanvänt (mer likt Keras än PyTorch), men PyTorchs `nn.Module` tillåter mer komplex och anpassningsbar logik (t.ex. if-satser mitt i framåtpasset). |
| **Lagertyper** | Tusentals (Transformers, 3D Conv, etc) | Dense, Conv2D, MaxPool, RNN, LSTM, BatchNorm, Dropout | TnsrLang täcker 99% av all klassisk Deep Learning, men saknar extrema specialfunktioner och transformer-primitiver (FlashAttention). |

## 2. Prestanda-rankning (0 - 10)

* **PyTorch (10/10)**: Optimerat med ATen (C++ backend), cuDNN för Nvidia GPU:er, MPS för Mac, och distribuerad träning.
* **TnsrLang (4/10)**: Använder Apple Accelerate (`cblas_dgemm`), vilket ger extremt snabb matris-multiplikation på Mac-processorer. För att vara ett hempulat ramverk är det otroligt snabbt, men det lider av stora överhettningar gällande minne.

---

## Potentiella Prestanda-optimeringar (Effektivisering)

Om du vill ta TnsrLang till nästa nivå och få det att springa extremt snabbt (närmare Caffe/PyTorch CPU-nivå), är det följande saker i C++-motorn som vi borde effektivisera:

### 1. Minnesallokering (Kritisk Flaskhals)
* **Problemet:** Just nu gör varje liten operation (`A + B`, `Sigmoid(A)`) en `new TensorImpl` och allokerar en helt ny `std::vector<double>` för både data och gradienter. Minnesallokeringar (`malloc/new`) är extremt långsamma i en träningsloop.
* **Lösningen:** Implementera en **Memory Pool** (Arena Allocator) i `MathEngine`. Istället för att be OS:et om minne varje gång, lånar man ut minne från en gigantisk förallokerad buffer.

### 2. Operator Fusion (Sammanslagning)
* **Problemet:** När vi gör `h = Tanh(DotProduct(x_t, W_x) + DotProduct(h, W_h) + b_h)` i LSTM-lagret, skapas 5 separata mellantensorer per tidssteg som sedan slängs bort.
* **Lösningen:** Skapa dedikerade C++-funktioner som `LSTMCellForward` som utför hela ekvationen i ett enda svep (en enda for-loop i C++) utan att allokera minne för mellanstegen.

### 3. Multi-threading för Element-wise operationer
* **Problemet:** Matris-multiplikation (`DotProduct` och `Conv2D`) är snabbt för att Apple Accelerate utnyttjar alla CPU-kärnor automatiskt. Men funktioner som `Sigmoid`, `ReLU` och `+` använder bara en klassisk `for`-loop på en enda kärna.
* **Lösningen:** Använd Apple GCD (`dispatch_apply`) eller OpenMP (`#pragma omp parallel for`) i `MathEngine.cpp` för att parallellisera looparna över vektorer.

### 4. AutoGrad Stack Overflow (Rekursion)
* **Problemet:** TnsrLangs `build_topo` (Topological Sort) i `backward()` är rekursiv. Om man tränar ett LSTM-nätverk med en sekvens på 10 000 tidssteg, kommer C++ krascha på grund av Stack Overflow.
* **Lösningen:** Skriv om topologisk sortering till att använda en iterativ loop (en stack/queue) istället för rekursion.

### 5. Uppdatera Apple Accelerate (BLAS)
* **Problemet:** Vid kompilering får vi varningar om att `cblas_dgemm` är deprecated i moderna macOS. 
* **Lösningen:** Lägg till kompilatorflaggan `-DACCELERATE_NEW_LAPACK` och anpassa datatyperna i `MathEngine.cpp` för att utnyttja Apples senaste ILP64-optimerade bibliotek.

---

## Sammanfattning
Språket är nu extremt imponerande arkitektoniskt. Funktionellt sett *har* du byggt en fungerande miniversion av PyTorch. För att närma sig i ren prestanda (execution speed) handlar nästa steg uteslutande om minneshantering i C++ (Memory Pooling och Operator Fusion).
