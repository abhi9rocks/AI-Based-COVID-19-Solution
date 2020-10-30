import cv2
import numpy as np
import tensorflow as tf
cap = cv2.VideoCapture(0)
cv2.namedWindow ('Dashboard', cv2. WINDOW_NORMAL)
cv2.setWindowProperty ('Dashboard', cv2. WND_PROP_FULLSCREEN, cv2. WINDOW_FULLSCREEN)
classes = ["unknown","mask","no_mask"]
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.gfile.FastGFile('xyz.pb','rb') as f:
	graph_def=tf.GraphDef()
	graph_def.ParseFromString(f.read())
with tf.Session() as sess:
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
	while (True):
		_, img = cap.read()
		rows=img.shape[0]
		cols=img.shape[1]
		inp=cv2.resize(img,(220,220))
		inp=inp[:,:,[2,1,0]]
		out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
				sess.graph.get_tensor_by_name('detection_scores:0'),
                      		sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		sess.graph.get_tensor_by_name('detection_classes:0')],
                     		feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
		num_detections=int(out[0][0])
	
		for i in range(num_detections):
			classId = int(out[3][0][i])
			score=float(out[1][0][i])
			bbox=[float(v) for v in out[2][0][i]]
			label=classes[classId]
			mask = 0 
			no_mask = 0
			if(classId == 2):
				no_mask = no_mask + 1
				color = (0,0,255)
			else:
				color = (0,255,0)
				mask = mask + 1
			

			if (score>0.5):
				x=bbox[1]*cols
				y=bbox[0]*rows
				right=bbox[3]*cols
				bottom=bbox[2]*rows
				
				
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=1)
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				cv2.putText(img, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
				cv2.putText(img, "MASK : "+str(int(mask)), (50, 440),1 , 1, (255, 0, 0), 2, cv2.LINE_4)
				cv2.putText(img, "NO_MASK : "+str(int(no_mask)), (50, 460),1 , 1, (255, 0, 0), 2, cv2.LINE_4)
		cv2.imshow('Dashboard',img)
		key=cv2.waitKey(1)
		if (key == 27):
			break
cap.release()
cv2.destroyAllWindows()