import tensorflow as tf

sess = tf.InteractiveSession()
input_batch = tf.constant([
    [   #第一张图片：2个像素
    [[0, 255, 0],[0, 255, 1],[0, 255, 2]] ,
    [[1, 255, 1],[1, 255, 1],[1, 255, 2]]
    ],
    [ #第二张图片：2个像素
    [[1, 255, 0],[1, 255, 0],[1, 255, 0]] ,
    [[255, 0, 0],[255, 0, 0],[255, 0, 0]]
    ]
])

sess.run(input_batch)
print(input_batch.get_shape())
sess.close()
