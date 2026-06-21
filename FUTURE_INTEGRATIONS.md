# TnsrLang: Future Integrations & Scaling Roadmap

Detta dokument fungerar som en vision för framtiden. Eftersom vi nyligen slutförde **Arkitektur-rensningen (v4.1)** där all matematik frikopplades från kompilatorn och Tensor-kärnan gjordes N-dimensionell, är grunden nu perfekt lagd. Nästa evolutionära steg är att anpassa språket för extrem hastighet, kompatibilitet och skala.

---

## 1. GPU Acceleration (Det Stora Klivet)

**Mål:** Byta ut vår C++ `for`-loop baserade MathEngine mot anrop till GPU-accelererade bibliotek.
När vi når detta steg kommer TnsrLang att sluta vara ett utbildningsverktyg och i stället kunna prestera i industriell hastighet.

### Teknologiska Val:
- **CUDA (Nvidia):** Det industriella standardvalet för AI. Vi kan byta ut våra tensor-uträkningar mot `cuBLAS` (CUDA Basic Linear Algebra Subprograms).
- **OpenCL / Metal:** Bättre för cross-platform stöd (exempelvis på Mac/Apple Silicon).

### Genomförande:
- Istället för att `TensorImpl` har en `std::vector<double> data`, kommer den att ha en pekare till ett minnesblock på grafikkortet (`cudaMalloc`).
- Våra funktioner (som `DotProduct`) exekverar en kernel på GPU:n istället för en C++ loop.

## 2. Avancerad AI (Bildigenkänning & CNNs)

**Mål:** Nu när TnsrLang stöder N-dimensionella tensorer kan vi bygga in stöd för bildbehandling (2D och 3D data).
Genom att skriva en Native Function för **Conv2D** (Convolutional Neural Networks) kan vi träna TnsrLang att "se". Nästa naturliga milstolpe för språket är att skriva ett TnsrLang-skript som laddar ner **MNIST-datasetet** och lär sig att identifiera handskrivna siffror. Detta är det ultimata beviset på ett moget AI-ramverk.

---

## 3. Python Bindings & Integreringar (Möjlig framtid)

**Mål:** Göra motorn tillgänglig från andra, högre programmeringsspråk (som Python), precis som PyTorch och TensorFlow gör i bakgrunden.

### Varför Python?
Nästan all AI-utveckling och data science görs i Python. Genom att binda vår kod kan vi skriva `import tnsrlang` i ett Python-script och använda den extremt snabba C++-motorn i bakgrunden.

### Genomförande:
- **pybind11:** Ett fantastiskt verktyg för att knyta samman C++ och Python.
- Vi exponerar `Tensor`-klassen via pybind11, så att Python kan manipulera våra tensorer, men AutoGrad-kedjan skapas blixtsnabbt i C++.

---

## 3. Inbyggd Spel- och App-motor

**Mål:** Att tillhandahålla TnsrLang som en "Single-header" eller lättviktig `.dll`/`.so` modul för spelutvecklare.

### Koncept:
När man bygger ett spel i C++ (ex. Unreal Engine) är det ofta omöjligt eller fruktansvärt tungt att lägga in ett massivt 4GB-bibliotek som TensorFlow bara för att ha lite AI. 
Vår optimerade `MathEngine` och `Evaluator` kan packas ner till en pytteliten C++-komponent som ger spelutvecklare en fullfjädrad maskininlärningsmotor på några kilobyte. 

## Sammanfattning
Med den nyligen slutförda arkitektur-uppdateringen och dessa kommande integrationssteg, går TnsrLang från att vara ett utbildningsprojekt till att bli en respekterad utmanare inom "Micro-frameworks" för AI. Vi bygger ett ekosystem som kombinerar PyTorchs kraft med en brutalistisk, glasklar design.
