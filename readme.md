##Project description

```
The purpose of this project is create a multi threading software to recognize weapons on 
videos or images. The method implemented to do the project workload is split video into 
number of available computer cores to assign each splited video to each core to generate
a folder with images then each available core will take a group of generated images to
analyze if has weapons inside. 
```

##Guide to use

###Used libraries
```
Library:            Use:

Movie py:           To split videos into images
Open-cv python:     To images recognition
time:               To get a delta of time to calculate duration 
concurrent:         To lauch and manage threads
multiprocessing:    To get a number of cores
argparse:           To use a python script launch arguments
numpy:              Mathematical library to process matrix
```
###Notes
```
To run python script from CMD or bash remeber that you needs to have installed python path
on windows case or create an symlink on linux case.
```

###Installation

```
To intall all needed libraries only execute:
pip install -r .\requirements.txt
```

###Parameters
```
video_path: Is where is located the video to process.

Example: --video_path C:\\Users\\allen\\Desktop\\example.3gpp```

Is passed like arguments on python execute
```
###Way to run

```
python .\main.py --video_path C:\\Users\\allen\\Desktop\\example.3gpp

Note: if video path is no sended will use the example video inside /videos folder.

python .\main.py 
```

###Code explanation 

```python
    def get_video_duration(video_clip_path):
        duration = VideoFileClip(video_clip_path).duration
        print(f"Duration: {duration}")
        return duration
```
>La funcion get_video_duration lo que hace es obtener la duracion total del video que se le pasa por parametro


```python
    def split_video(video_clip_path, init_time, end_time, portion):
        result = VideoFileClip(video_clip_path).subclip(init_time, end_time)
        result.write_videofile(f"spliced_videos/{portion}.mp4", threads=multiprocessing.cpu_count(), audio=False)
        result.close()
```
>La funcion split_video se encarga de partir un video en los parametros enviados init_time y end_time, la porcion es 
>un identificador para determinar el nombre de cada sub video

```python
    def split_video_multiprocessing(video_clip_path):
        slices = multiprocessing.cpu_count()
        slides_portion = get_video_duration(video_clip_path) / slices
        print(f"Portions: {slides_portion}")
        print(f"slices: {slices}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for x in range(0, slices):
                print(f"Init time: {x * slides_portion} End time: {(x + 1) * slides_portion} Name: {x}")
                result = executor.submit(split_video, video_clip_path, x * slides_portion, (x + 1) * slides_portion, x)
```
>La funcion split_video_multiprocessing se encarga de asignar los recursos de los hilos para llamar las funcion split_video en cada porcion determinada

```python
    def get_frame(sec, video_capture, thread, number):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frame, image = video_capture.read()
        if has_frame:
            cv2.imwrite(f"images_/{thread}{str(number)}.jpg", image)
        return has_frame
```

>La funcion get_frame lo que hace es extraer imagenes de un momento en especifico del video

```python
    def video_to_images(video_clip_path, thread):
        video_capture = cv2.VideoCapture(video_clip_path)
        sec = 0
        frame_rate = 1
        count = 0
        success = get_frame(sec, video_capture, thread, count)
        while success:
            count += 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success = get_frame(sec, video_capture, thread, count)
        return success
```

>La funcion video_to_images obtiene el sub video previamente partido y usa la funcion get_frame para ir obteniendo frame a frame las imagenes, el parametro thread es un nombre identificador para saber quien lo esta procesando

```python
    def splice_video():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for filename in os.listdir("spliced_videos"):
                executor.submit(video_to_images, f"spliced_videos/{filename}", f"THREAD{filename}-")
```

>La funcion splice_video se encarga de asignar los recursos de los hilos para cada video a convertir en imagenes

```python
    def load_yolo():
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
    
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = numpy.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers
```
> La funcion load_yolo se encarga de cargar todo lo necesario para el reconocimiento de objetos, como el modelo entreando

```python
    def load_image(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        return img, height, width, channels
```

>La funcion load_image se encarga de preparar la imagen para el reconocimiento de objecto, carga la images, y le aplica un reajuste de tamaño

```python
    def detect_objects(img, net, output_layers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        return blob, outputs
```

>La funcion detect_objects aplica la descomposicion de la imagen para aplicar las redes neuronales de reconocimiento de imagenes y retorna un objeto que tiene los datos de si se encontro algun objeto que calza con lo que se esta buscando

```python
    def get_box_dimensions(outputs, height, width):
        boxes = []
        configurations = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = numpy.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    configurations.append(float(conf))
                    class_ids.append(class_id)
        return boxes, configurations, class_ids
```
>La funcion get_box_dimensions se encarga de crear los cuadros que delimitaran los objetos encontrados

```python
    def draw_object_labels(boxes, configurations, colors, class_ids, classes, img, img_name, number):
        founded = False
        indexes = cv2.dnn.NMSBoxes(boxes, configurations, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                founded = True
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
    
                color = 255
                if i < len(colors):
                    color = colors[i]
    
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
        img = cv2.resize(img, (800, 600))
        if founded:
            cv2.imwrite(f"images_processed/{img_name}_{number}.jpg", img)
```
>Se encarga de dibujar los cuadros sobre los objetos encontrados y en caso de que contenga un objeto guarda la imagen

```python
    def image_detection(img_path, number):
        model, classes, colors, output_layers = load_yolo()
        image, height, width, channels = load_image(img_path)
        blob, outputs = detect_objects(image, model, output_layers)
        boxes, configurations, class_ids = get_box_dimensions(outputs, height, width)
        draw_object_labels(boxes, configurations, colors, class_ids, classes, image, img_path.split("/")[1], number)
```

>La funcion image_detection llama a todas la funciones relacionadas con el yolo para el reconocimiento de objetos

```python
    def detection_by_thread(range_init, range_final):
        print(f"Init: {range_init} : End: {range_final}")
        counter = 0
        for filename in os.listdir("images_"):
            if range_init <= counter <= range_final:
                image_detection(f"images_/{filename}", counter)
            counter += 1
```
>La funcion detection_by_thread llama al reconocimiento de imagenes en un bloque segun la cantidad de nucleos disponibles


```python
    def run_yolo_detection():
        **images_quantity** = 0
    
        for filename in os.listdir("images_"):
            images_quantity += 1
    
        images_by_thread = images_quantity / multiprocessing.cpu_count()
        images_by_thread_rounded = round(images_by_thread, 0)
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for x in range(0, multiprocessing.cpu_count()):
                if x * images_by_thread_rounded + images_by_thread_rounded >= images_quantity:
                    executor.submit(detection_by_thread, x * images_by_thread_rounded, images_quantity)
                else:
                    executor.submit(detection_by_thread, x * images_by_thread_rounded,
                                    x * images_by_thread_rounded + images_by_thread_rounded)
```
>La funcion run_yolo_detection se encarga de la gestion de los hilos por bloques de imagenes a procesar segun la cantidad de nucleos