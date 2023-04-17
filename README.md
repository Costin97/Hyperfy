# Hyperfy
Detection Models

Aplicatie in care vom aplica YoloV2 si SSD(Single Shot-Detector) asupra unui stream video.

*scurta descriere a ceea ce am incercat sa realizez*

yolov2_ssd_methods.py

Initial dam import la modulele necesare realizarii programului (in cazul de fata cv2)
Definim clasa VideoCapture,in cadrul careia,in metoda __init__,ce joaca rol de constructor
introducem parametrii "detector" (ce reprezinta instanta unei clase pe care o voi explica ulterior),
"url" (link-ul stream-ului video asupra caruia vom aplica cele doua modele de detectie a obiectelor),
"ssd_model_path" (path-ul catre fisierul ce contine modelul propriu-zis:trained weights and graph definition
) si "ssd_config_path" (path-ul catre configuration file:confine arhitectura si parametrii 
folosite de SSD) si nu in ultimul rand "labels_path" (fisier in cadrul caruia sunt definite)
Ulterior,initializam obiectele de captura video atat pentru Yolo cat si pentru
SSD,folosindu-ne de cv2.VideoCapture().Initializam si modelul SSD,incarcandu-l cu ajutorul functiei
cv2.dnn.readNetFromTensorflow(),care ia ca parametrii model_path,config_path.Folosindu-ne de 
labels_path extragem si elementele ce pot fi detectate (etichetele lor).
In cadrul metodei: ssd_method(),este aplicat modelul SSD asupra frame-ului corespunzator.
Pentru a putea fi aplicat,trebuie ca frame-ul sa fie convertit in format blob,cu ajutorul
lui cv2.dnn.blobFromImage().Ulterior acest frame convertit,este trimis ca si input catre model.
Acesta returneaza detectarile pe care a reusit sa le faca,pe care le parcurgem cu ajutorul
unui for loop,in cadrul caruia "desenam" si overlay-ul de dreptunghiuri pentru a putea sublinia
obiectele ce au fost gasite.
Ultima metoda,run(),citeste frame-urile din cadrul celor doua obiecte VideoCapture (cap1,cap2)
Frame-urile primesc un resize,cu ajutorul cv2.resize() (acest resize a fost facut strict din
considerente de incadrare pe ecran si de a arata oarecum estetic),inainte de a fi trimise celor doua modele
(YOLO si SSD).Chemand metoda detection_algorithm,din cadrul clasei YoloV2Model,primim inapoi
chenarele obiectelor pe care le-a detectat modelul YOLOV2.Parcurgand aceste chenare,desenam 
pe frame-ul atribuit acestui model dreptunghiuri,incadrand obiectele detectate.
In final concatenam cele 2 frame-uri cu ajutorul cv2.hconcat(),si frame-ul rezultat il
afisam prin intermediul cv2.imshow().Acest loop continuu se poate incheia prin apasarea tastei "q"
pe care am predefinit-o,(quit),si pe care o urmarim cu ajutorul:
            if cv2.waitKey(1) == ord('q'):
                return
Cand se incheie loop-ul,eliberam resursele asignate celor doua obiecte VideoCapture,cu ajutorul 
chemarii a doua metode self.cap1.release() si self.cap2.release().

yolov2_model.py

La fel ca mai sus,initial dam import la modulele necesare aplicarii modelului.Apoi
definim clasa YoloV2Model,pe care am mentinat-o si mai sus,care primeste doua argumente:"model_path"
ce este calea catre fisierul ce contine modelul yolov2,asemanator cu cazul de mai sus,si "weights_path"
ce contine parametrii antrenati ce se folosesc pentru detectare.
In cadrul constructorului,modelul este incarcat cu ajutorul:cv2.dnn.readNetFromDarknet(),
care primeste parametrii necesari (model_path,weights_path).Straturile output-ului sunt obtinute
prin getter-ul necesar,get_output_layers().Aceasta functie extrage layere-le care sunt folosite
pentru realizarea predictiilor in cadrul modelului YoloV2,generate cu ajutorul metodei self.model.getUnconnectedOutLayers().
Functia detection_algorithm(),primeste frame-ul necesar si aplica modelul asupra sa.
Ca si mai sus trebuie modificat frame-ul si transformat in format blob.Acest blob optinut este trimis
ca input modelului,si ulterior este obtinuta o lista de Numpy arrays,pentru fiecare output layer,de mai sus,
arrays,ce contin "detectarile" facute de model.
Chenarele cu confidence ridicat sunt extrase(intre timp se aplica diverse formule asupra
parametrilor) si returnate ca o lista de tuples,sub forma (x,y,width,height),lista ce este folosita,asa cum am descris mai sus,
pentru a incadra obiectele detectate.
	

ssd_model_blueprint.py

Aceasta clasa a fost realizata cu scopul de a aplica modelul SSD (Single Shot-Detector)
asupra stream-ului video,insa ulterior mi-am dat seama ca o solutie pentru a putea rezolva cerintele 
2a,2b,2c ar fi sa introduc aceste linii de cod in zona in care s-ar aplica si modelul YoloV2,asa ca am pastrat clasa 
strict cu scop informativ,nicio metoda de aici nu este utilizata propriu-zis in cadrul programului.

De mentionat este faptul ca in cadrul fisierului yolo_model.py,exista un array de urls.In cazul in care se doreste testarea
programului propriu-zisa am lasat acolo mai multe link-uri de camera ce pot fi utilizate,intrucat exista momente ale zilei,in
care intr-o zona este seara si nu se vede mai nimic,ceata etc.
