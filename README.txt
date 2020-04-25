=========================================Strojové učení a rozpoznávání-SUR==========================================
Projekt: Detektor jedné osoby z obrázku obličeje a hlasové nahrávky.

Systém: Klasifikátor audia pomocou neurónovej siete v keras API
Výsledky: audio_NN_MLP
Súbory: audio_classifier_MLP.ipynb
            - jupyter notebook, v ktorom bola neurónová sieť natrénovaná na trénovacích dátach poskytnutých v zadaní. Z
            dôvodu rôznorodosti váh pri trénovaní neurónovej siete som natrénovanú sieť uložil separátne aby bolo možné
            overenie výsledkov klasifikovania. Natrénovaný model je uložený v zložke 'audio_classifier.h5'.
        audio_classifier_MLP.py
            - spracováva zložku(eval) a zapisuje výsledky už na základe predom natrénovaného modelu, ktorý je separátne
            uložený v súbore 'audio_classifier.h5'. Demonštračný súbor funkčnosti klasifikátora.
Verzia pythonu pri implementácií: 3.6.10
Potrebné API a knižnice:
	h5py - uloženie a načítanie natrénovaných tensorflow modelov
			pip install -q pyyaml h5py
	librosa - slúži na spracovávanie audia
			pip install librosa
	pandas - služí na čítanie dát z csv súbora
			pip install pandas
	keras - Neural network API 
			pip install keras
	matplotlib - Pokročilé vizualizácie dát
			pip install matplotlib
	numpy - Pokročilé výpočty
			pip install numpy
	numba v0.48 - Scientific computing
			pip install numba=0.48.0
Popis: Z wav súborov obsahujúcich nahrávky osôb extrahujem príznaky ako MFCC, Zero-crossing-rate, Spectral-rollof a
ďalšie.. Vyextrahované príznaky si uložím do csv súboru v ktorom okrem príznakov evidujem aj cestu k wav súboru a názov
osoby. Z csv súbora vyberám príznaky oddelene s názvom súborov/osôb a vytváram pole labelov, v ktorom hľadanú osobu
označujem ako 1 inak 0. Na týchto príznakoch a labeloch trénujem neurónovú sieť, vďaka ktorej predikujem či sa jedná
o hľadanú osobu. Neurónová sieť sa skladá zo 6 Dense layerov predstavujúci Multi Layer Perceptron(MLP).

===================================================================================================================
