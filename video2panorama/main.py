import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from utils import most_common_element, get_movement_direction_AND_speed, fill, trim_black_rows_or_columns

#####
# 1) Alegere hyoerparametrii
#####

feature_extraction_algo = 'sift' # se permite doar sift si orb. Se recomanda sift
#*SCHIMB* - acest ratio ar trebui reglat automat sau mutat sus la parametrii
ratio = 0.75
#*SCHIMB* - ar trebui calculat dinamic in fucntie d edimensiunea imaginii
#reprojectionThresh = 4
reprojectionThresh = 2
adaptive_shift_pixels = 2
# acesta este numarul pixelilor pe care poate varia imaginea pe directia nedomiannta 
# - o valoare mica tinde sa pastreze formatul dreptunghiular al imaignii, air una mare creeaza forme nereglate, DAR pastreaza mai mult din perspectiva
numar_pixeli_grad_libertate_directie_nedominanta = 0

#begin_idx = 10
#end_idx = 80
#hop = 4

##################################### Load img
#####
# *) Acest pas este doar pentru incarcarea imaginilor in cazul in care fac pe imagini nu pe video
#####

# load the video
video_path = r'data\video_horiz_sd.mp4'
#video_path = r'data\video_horiz_ds.mp4'
#video_path = r'data\video_vert_sj.mp4'
#video_path = r'data\video_vert_js.mp4'


######
# 0) Detectare sens miscare
######

# Replace 'path_to_video' with your video file path
direction, speed = get_movement_direction_AND_speed(video_path, 4)
print('Directia miscarii este: ', direction)
print('Viteza miscarii, ca medie a magnitudinii vectorilor flux optic, este: ', speed)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# vreau sa aflu numarul de frameuri
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Numar total de cadre: : ', total_frames)

# se incearca adaptarea parametrilor la nr de frameuri
begin_idx = min(10, total_frames)
end_idx = total_frames
#hop = 8
# adaptive hop - a fost setat experimetal
# cand viteza creste trebuie ca hopul sa scada
adaptive_hop = round(164/speed) # coeficientul 164 a fost ales bazat pe experimente realizate la diferite vitreze
hop = adaptive_hop
print('Hopul adaptiv calculat este: ', adaptive_hop)

# inversez videoclipul 
if direction == 'ds' or direction == 'js':
    buffer = end_idx
    end_idx = begin_idx
    begin_idx = buffer  -1
    hop = -hop

cap.set(cv2.CAP_PROP_POS_FRAMES, begin_idx)
# ret va fi boolean - inidca daca s-a returnat sau nu ceva; frame va fi cadrul retunat
ret1, frame1 = cap.read()
query_photo = frame1
query_photo = cv2.cvtColor(query_photo,cv2.COLOR_BGR2RGB)
query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)
begin_idx = begin_idx + 2 * hop

for i, idx in enumerate(tqdm(range(begin_idx, end_idx, hop))):

    #train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    # ret va fi boolean - inidc
    ret2, frame2 = cap.read()
    train_photo = frame2
    train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

    if direction == 'js' or direction == 'sj':
        # camera mea din apcate adauga doua linii jos care sunt negre, dar ar nu negrre 100% - artefact camera - problema specifica
        train_photo = np.copy(train_photo[:-2, :, :])
        query_photo = np.copy(query_photo[:-2, :, :])

    #####
    # 2) Selectarea tipului de desciptori conform hyperparametrilor. *SCHIMB* - probabil o s alas doar un descriptor la deployment.
    #####

    # selectezmetoda de extragere a trasaturilor pentru crearea descriptorilor
    if feature_extraction_algo == 'sift':
        descriptor = cv2.SIFT_create()
    elif feature_extraction_algo == 'orb':
        descriptor = cv2.ORB_create()

    # se creeaza practic doua lite: una cu coordonatele punctelor si alta cudescriptorii asociati punctelor. Cele doua liste au o corespondenta dpdv al ordinii
    # extragerea propriu-zisa a trasaturilor cu metoda specificata ca hyperparametryu
    #print(train_photo_gray.shape)
    #print(query_photo_gray.shape)
    keypoints_train_img, features_train_img = descriptor.detectAndCompute(train_photo_gray, None)
    keypoints_query_img, features_query_img = descriptor.detectAndCompute(query_photo_gray, None)
    # fiecare vector de trasaturi are dimensiune de 128 pt SIFT si 64 pt SURF, de exemplu
    # un obiect KeyPoint are -> x, y (cordonatlee), size, angle, response, octave (octava e rezolutia in sistemul de octave-piramida)
    
    
    '''
    Daca folsoesc SIFT (128 e lungimea pt descriptori)
    - in vecinatatea fiecarui punct se alege o fereastra de 16x16
    - ferastar de 16x16 se imparte in 4x4 sub-blocuri de size 4x4
    - pe fiecare sub-bloc se calculeaza histograma orietarilor, cu o cuantizare de 45n degrade -> adica o valori
    - de aici rezulta 4x4 * 8 valori = 128
    - se calculeaza gradientii (un gradient are 1.MAGNITUDINE si 2.ORIENTARE) in fiecre punct (cum faceam la AI, de ex, cu sobe-uri sau ceva asemanatori)
    - apoi se obtine HoG calculand PENTRU FIECRAE ORIENTARE - acumularea magnitudinilor presupun ca se face ca la AI cu masti 
    - ulterior obtinerii gradientilor, acestai se normalizeaza penrua  obtine invarianta luminant a(iluminarea scenei)
    - CONCLUZIE: folosec HoG
    '''


    '''
    V1) Brute force (am scos-o ca  proasta)
    Algoritm: se compara fiecare vector de trasaturi dintr-un set cu toate celelelate trasaturi din celalalt set si se compara pe baza dinstantei euclidiene, daca folosesc SIFT sau pe baza distantei hamming, daca folosesc ORB. 
    PROBLEMA: pot exista pattern-uri repetitive care sa cuzeze un ‘best match’ intre niste puncte care, de fapt, nu corespund acelarasi puncte, in realtitate. 
    Solutia: KNN 

    V2) KNN cu k = 2 si test lowe
    # Se aplica TESTUL DE RATIE al lui LOWE's -> se verifica daca raportul distantelor dintre 2 desciptori este sub un anumit prag
    # cel_mai_apropiat/al_doilea_cel_mai_apropiat < ratie
    # Logica: ar trebui sa existe o difernta semnificativa intre best match si a l doilea. Daca nu, probabil ele provin dintr-un pattern repetitiv
            
    '''

    # instantiere obiect care face potrivirea
    if feature_extraction_algo == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif feature_extraction_algo == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # compute the raw matches and initialize the list of actual matches
    # asta va fi o lista de potriviri peste care voi itera
    # se retuneaza cei mai apropiati k vecini, din care se va alege cea mai buna potrivire a.i. sa respecte critetriul lui Lowe
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    # noua lista, in care voi pune matchurile cele mai bune
    matches = []

    # iterare prin potriviri pt a le filtra si a le pastra doar pe cele credibile (pe baza TESTULUI DE RATIE al lui LOWE's)
    for cel_mai_apropiat,al_doilea_cel_mai_apropiat in rawMatches:
        # Se aplica TESTUL DE RATIE al lui LOWE's -> se verifica daca raportul distantelor dintre 2 desciptori este sub un anumit prag
        # cel_mai_apropiat/al_doilea_cel_mai_apropiat < ratie
        # Logica: ar trebui sa existe o difernta semnificativa intre best match si a l doilea. Daca nu, probabil ele provin dintr-un pattern repetitiv
        if cel_mai_apropiat.distance/al_doilea_cel_mai_apropiat.distance < ratio:
            matches.append(cel_mai_apropiat)

    # afisez 100 de matchuri
    mapped_features_image_knn = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img, np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    ######
    # 3) Se calculeaza matricea de omografie 
    ######


    # transform tuplurile care dau coordonatele in ndarray, und efiecare linie reprezinta o coordonata
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    # Alg. RANSAC nu poate fi aplicat decat daca exista minim 4 puncte comune
    if len(matches) > 4:
        # se transforma in ndarray punctele caracteristice dintre imagini ca sa fie acceptate de cv2.findHomography
        # acum se selecteaza acele puncte care se potrives, din ndarrayul pe puncte anterior format
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        
        # folosirea alg RANSAC pt aflarea matricei de omografie
        # reprojThresh controleaza care puncte sunt considerate inliner - ||pi - pi'|| < reprojThresh, unde pi' este proeictia lui pi in noul sistem (daca e mai mic e inliner)
        # *SCHIMB* - aici voi face un calcul al reprojThresh relativ la dimensiunea imaginiii - tre sa gasesc o formula
        (Homography_Matrix, _) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojectionThresh)
    else:
        # *SCHIMB* - la elese pun continue spre urmatoarea bucla
        print("Nu s-a putut calcula matricea de omografie.")
        continue
    if Homography_Matrix is None:
        print("Nu s-a putut calcula matricea de omografie.")
        continue
    ######
    # 4) Lipirea/combinarea imaginilor
    ######

    # calcul dimensiune imagine de output

    # ***************** urnatoarele 2 linii ;le modific pentru vericala *********************8

    # impartirea dupa directii
    if direction == 'sd' or direction == 'ds': # daca miscarea e pe orizonatla
        # sumez latimile imaginilor
        width = query_photo.shape[1] + train_photo.shape[1]

        # pentru inaltime iau maximul ianltimilor
        height = max(query_photo.shape[0], train_photo.shape[0])
        # %%%%%%%%%%%%% MODIFICAT
        height += numar_pixeli_grad_libertate_directie_nedominanta
        # %%%%%%%%%%%%% MODIFICAT

        # aici se aplica transformarea de perspectiva pe imaginea a doua -> practic se face transformarea omografica
        # imaginea rezultata va fi imaginea care se adauga (cadrul urmator) transformata pe noiel coordonate
        result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))

        # se pastreaza a doua imagine transformata, peste care se adauga, la inceput, prima imaggine originala nealterata
        #result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo
        ##### PROBLEMA CODULUI COMENTAT DE MAI SUS:
        # Probelam e ca cadrul anterior can se suprapune peste veriunea distorsioanata
        # a primului cadru suprascrie din el, pt ca e negru.
        # SOLUTIA: creez o masca binara ca sa nu se copieze peste iamginea modificata decat pixelii diferiti de zero.
        # shiftez la dreaptra masca si o adun cu ea insasi - asa sper sa elimin artefactul
        mask = np.any(query_photo != 0, axis=2)
        # In mod adpativ, numarul de pixlei shiftati crste.
        if i % 5 == 0:
            adaptive_shift_pixels += 1
        extended_mask = np.hstack((mask[:, adaptive_shift_pixels:], np.ones((mask.shape[0], adaptive_shift_pixels), dtype=bool)))
        extended_mask = fill(extended_mask)
        extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)
        

        # Apply the mask to copy only non-zero pixels from query_photo to the corresponding position in result
        result[0:query_photo.shape[0], 0:query_photo.shape[1]][extended_mask_3d] = query_photo[extended_mask_3d]
    elif direction == 'sj' or direction == 'js': # daca miscarea e pe vericala
        # sumez latimile imaginilor
        width = max(query_photo.shape[1], train_photo.shape[1])
        # %%%%%%%%%%%%% MODIFICAT
        width += numar_pixeli_grad_libertate_directie_nedominanta
        # %%%%%%%%%%%%% MODIFICAT

        # pentru inaltime iau maximul ianltimilor
        height = query_photo.shape[0] + train_photo.shape[0]
        '''
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(query_photo)
        plt.show()

        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(train_photo)
        plt.show()
        '''

        # aici se aplica transformarea de perspectiva pe imaginea a doua -> practic se face transformarea omografica
        # imaginea rezultata va fi imaginea care se adauga (cadrul urmator) transformata pe noiel coordonate
        result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))
        '''
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(result)
        plt.show()
        '''

        mask = np.any(query_photo != 0, axis=2)
        # In mod adpativ, numarul de pixlei shiftati crste.
        if i % 5 == 0:
            adaptive_shift_pixels += 1
        '''
        extended_mask = np.hstack((mask[:, adaptive_shift_pixels:], np.ones((mask.shape[0], adaptive_shift_pixels), dtype=bool)))
        extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)
        '''
        
        ''' # apropae merge
        extended_mask = np.vstack((mask[adaptive_shift_pixels:, :], np.ones((adaptive_shift_pixels, mask.shape[1]), dtype=bool)))
        extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)
        '''
        extended_mask = np.vstack((mask[adaptive_shift_pixels:, :], np.ones((adaptive_shift_pixels, mask.shape[1]), dtype=bool)))
        extended_mask = fill(extended_mask)
        extended_mask_3d = np.repeat(extended_mask[:, :, np.newaxis], 3, axis=2)

        # Apply the mask to copy only non-zero pixels from query_photo to the corresponding position in result
        result[0:query_photo.shape[0], 0:query_photo.shape[1]][extended_mask_3d] = query_photo[extended_mask_3d]

    else:
        print('Problema la estimarea miscarii. trebuia ca direction sa feie ori js ori sj ori ds ori sd')
    # Pregatesc variabilele pentru bucla urmatoare
    query_photo = np.copy(result)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

# elimin coloanele si liniile negre
if direction == 'ds' or direction== 'sd':
    query_photo = trim_black_rows_or_columns(query_photo, 'col')
else:
    query_photo = trim_black_rows_or_columns(query_photo, 'row')

# %%%%%%%%%%%%% MODIFICAT
query_photo = trim_black_rows_or_columns(query_photo, 'col')
query_photo = trim_black_rows_or_columns(query_photo, 'row')
# %%%%%%%%%%%%% MODIFICAT

plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(query_photo)
plt.show()