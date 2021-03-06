Static Gesture Recognition using Leap motion
==

**An automated bar ordering system for making orders in a bar using hand gestures.**

> The purpose of this project was to develop an ordering system for a bar where orders were made only by using static hand gestures. 
With the help of a Leap Motion controller, we used Machine Learning to train a model to recognize 8 different hand gestures, in which the user could (with the support of the UI) navigate the ordering system ordering any number of drinks, foods and even to select payment option solely by using hand gestures.

Paper available [here](https://arxiv.org/abs/1705.05884).

## Features
* Static hand gesture recognition.
* Voice and written response to different gestures.
* Simple to add new gestures.

## Requirements
- A [Leap Motion.](https://www.leapmotion.com/)

## Run
`$ py mainProgram.py`

## Training and saving gesture data (optional)

### Save gestures
Save each gesture multiple times using: `$ py saveGesturesRaw.py #`
Where `#` is an integer representing each gesture.

### Train
`$ py training/training.py`

## Flow
Flow chart of the UI. The user interacts using gestures and receives both
written and spoken response.
![OrderingFLow](https://cvml1.files.wordpress.com/2016/05/basic-flow-chart.png?w=600)

## Gestures
![GestureGIF](https://cvml1.files.wordpress.com/2016/05/output_ttzsia.gif?w=900)

- Init.
- Alcoholic drink.
- Non alcoholic drink.
- Food.
- Undo.
- Checkout.
- Pay with cash.
- Pay with credit card.

## Blog
See our blog from the project [here.](https://cvml1.wordpress.com/)

## Report
[Project report.](https://github.com/windmark/static-gesture-recognition/blob/master/report.pdf)

## Team
[![Christofer Lind](https://avatars0.githubusercontent.com/u/5421089?v=3&s=144)](https://github.com/chilind)
[![Maria Svensson](https://avatars2.githubusercontent.com/u/5993475?v=3&s=144)](https://github.com/mariasvenson)
[![Babak Toghiani-Rizi](https://avatars2.githubusercontent.com/u/5991620?v=3&s=144)](https://github.com/babaktr)
[![Marcus Windmark](https://avatars0.githubusercontent.com/u/3810163?v=3&s=144)](https://github.com/windmark)

