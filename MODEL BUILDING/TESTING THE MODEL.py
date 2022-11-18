testingpath=list(paths.list_images(testingpath))
idxs=np.arange(0,len(testingpath))
idxs=np.random.choice(idxs,size=(25,),replace=False)
images=[]

for i in idxs:
    image = cv2.imread(testingpath[i])
    output = image.copy()

    # load the input image,convert to grayscale and resize

    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # quantify the image and make predictions based on the  extracted feature using last trained random forest
    features = quantify_image(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]
    # the set of output images
    if label == "healthy":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    images.append(output)

# creating a montage
montage = build_montages(images, (128, 128), (5, 5))[0]
cv2.imshow("Output", montage)
cv2.waitKey(0)