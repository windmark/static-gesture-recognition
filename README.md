Static Gesture Recognition
===
##_using Leap motion_

**An automated bartender system for making orders in a bar using hand gestures.**

> We used a Leap Motion to find coordinates of palms and fingertips for a users both hands.
Then we trained a model, using machine learning, to recognize eight static hand gestures.
These gestures, and a UI, allowed a user to order any amount of drinks or food, undo actions
and finally order using either cash or credit payment in a fictitious bar.

## Features
* Static hand gesture recognition.
* Voice and written response to different gestures.
* Simple to add new gestures.

## Requirements
- A [Leap Motion.](https://www.leapmotion.com/)
- Leap Motion SDK?

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

## Paper
Link to our paper?

## Team
[![Christofer Lind](https://avatars0.githubusercontent.com/u/5421089?v=3&s=144)](https://github.com/chilind) | [![Maria Svensson](https://avatars2.githubusercontent.com/u/5993475?v=3&s=144)](https://github.com/mariasvenson) | [![Babak Toghiani-Rizi](https://avatars2.githubusercontent.com/u/5991620?v=3&s=144)](https://github.com/babaktr) | [![Marcus Windmark](https://avatars0.githubusercontent.com/u/3810163?v=3&s=144)](https://github.com/windmark)
---|---|---
[Christofer Lind](https://github.com/chilind) | [Maria Svensson](https://github.com/mariasvenson) | [Babak Toghiani-Rizi](https://github.com/babaktr) | [Marcus Windmark](https://github.com/windmark)

## License
License?
