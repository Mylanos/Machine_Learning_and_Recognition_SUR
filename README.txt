=========================================Strojové učení a rozpoznávání-SUR==========================================
Projekt: Detektor jedné osoby z obrázku obličeje a hlasové nahrávky.
Operačný systém : Ubuntu 19.10 , MacOs Catalina
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

Systém: Klasifikátor obličeje hladanej osoby ( Lineárni klasifikátor )
Výsledky: image_linear.txt
Súbory: image_classifier_evaluation.py
			 -- načíta klasifikátor z config.json "image_classifier" ( priečinok models , image classifier). 
			 -- prechádza obrázky z daného priečinka ( data/eval ) . Z každého obrázku si zistí hog features a dá klasifikátoru.
			 -- Klasifikátor uloží výsledky do dict table a vypíše dané výsledky do result image_classifier 


		image_classificator_training.py
			--otvorí každý png súbor  z priečinka určeným  v config.json  "non_target_train_f" , "target_train_f".
			--každému obrázku nastaví rovnakú veľkosť, otočí ho do 45 stupnov dolava aj do prava  , prekovertuje ho na gray farbu z cv2 a pomocou imutils zvýrazni hrany
			--Potom každý takto upravený obrázok ide do funkcie hog ktorá získa príznaky histogram of gradients.
			-- tieto príznaky idú do klasifikátora na trénovanie. Triedy sa určujú tak že non targetu sa priradia 0 a targetu sa priradia 1
			-- klasifikátor sa uloží poďla config.json "image_classifier". Defaultne ide do priečinku models


		Knižnice :
			- imutils
				- zvýraznenie hrán
			- matplotlib.pyplot 
				- vykreslenie hog a obrázku
			- PrettyTable
				-  ascii tabulka
			- skimage.feature.hog
				- Zbieranie hog features

Popis: Z png súborov obsahujúcich obrázky osôb extrahujem príznaky ako Gradienty( vektory v hog ) 
Vyextrahované príznaky si uložím do listu ktorý potom predávam na trénovanie obrázku. Tento klasifikátor potom prehľadáva data z eval a rozhoduje o nich či sú alebo niesú vyhľadávanou osobou.

Verzia pythonu pri implementácií: 3.6.10
===================================================================================================================
