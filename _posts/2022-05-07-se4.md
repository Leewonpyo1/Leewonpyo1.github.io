---
title:  "[2022810089] 이원표 7차과제-4"
excerpt: "md 파일에 마크다운 문법으로 작성하여 Github 원격 저장소에 업로드 해보자. 에디터는 Visual Studio code 사용! 로컬 서버에서 확인도 해보자. "

categories:
  - Blog
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2020-05-25
last_modified_at: 2020-05-25
---
# 요리 추천 웹 앱 구축

이 단원에서는 이전 단원에서 배운 몇 가지 기술과 이 시리즈 전체에서 사용된 맛있는 요리 데이터 세트를 사용하여 분류 모델을 구축합니다. 또한 Onnx의 웹 런타임을 활용하여 저장된 모델을 사용하는 작은 웹 앱을 빌드합니다.

머신 러닝의 가장 유용한 실제 용도 중 하나는 추천 시스템을 구축하는 것이며, 오늘 바로 그 방향으로 첫 걸음을 내딛을 수 있습니다!

[![Presenting this web app](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 동영상을 보려면 위 이미지를 클릭하세요. Jen Looper는 분류된 요리 데이터를 사용하여 웹 앱을 빌드합니다.

## [Pre-lecture quiz](https://white-water-09ec41f0f.azurestaticapps.net/quiz/25/)

이 단원에서는 다음을 배우게 됩니다.

- 모델을 빌드하고 Onnx 모델로 저장하는 방법
- Netron을 사용하여 모델을 검사하는 방법
- 추론을 위해 웹 앱에서 모델을 사용하는 방법

## 모델 구축

적용된 ML 시스템을 구축하는 것은 비즈니스 시스템에 이러한 기술을 활용하는 데 중요한 부분입니다. Onnx를 사용하여 웹 애플리케이션 내에서 모델을 사용할 수 있습니다(필요한 경우 오프라인 컨텍스트에서 사용).

In a [previous lesson](../../3-Web-App/1-Web-App/README.md), UFO 목격에 대한 회귀 모델을 만들고 "절임"하고 Flask 앱에서 사용했습니다. 이 아키텍처는 알고 있으면 매우 유용하지만 전체 스택 Python 앱이며 요구 사항에 JavaScript 응용 프로그램 사용이 포함될 수 있습니다.

이 단원에서는 추론을 위한 기본 JavaScript 기반 시스템을 구축할 수 있습니다. 그러나 먼저 모델을 훈련시키고 Onnx에서 사용할 수 있도록 변환해야 합니다.

## 훈련 분류 모델

먼저 우리가 사용한 정리된 요리 데이터 세트를 사용하여 분류 모델을 훈련시킵니다.


1. 유용한 라이브러리를 가져오는 것으로 시작합니다.

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    You need '[skl2onnx](https://onnx.ai/sklearn-onnx/)' to help convert your Scikit-learn model to Onnx format.


2. 그런 다음 `read_csv()`를 사용하여 CSV 파일을 읽어 이전 강의에서 했던 것과 같은 방식으로 데이터를 사용합니다

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

3. 처음 두 개의 불필요한 열을 제거하고 나머지 데이터를 'X'로 저장합니다.

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```


4. 레이블을 'y'로 저장합니다.

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### 훈련 루틴 시작


정확도가 좋은 'SVC' 라이브러리를 사용하겠습니다.


1. Scikit-learn에서 적절한 라이브러리를 가져옵니다.

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```


1. 훈련 세트와 테스트 세트를 분리합니다.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```


2. 이전 단원에서 했던 것처럼 SVC 분류 모델을 빌드합니다.

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

3. 이제 `predict()`를 호출하여 모델을 테스트합니다.

    ```python
    y_pred = model.predict(X_test)
    ```
4. 분류 보고서를 인쇄하여 모델의 품질을 확인합니다.

    ```python
    print(classification_report(y_test,y_pred))
    ```

    
이전에 보았듯이 정확도는 좋습니다.

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### 모델을 Onnx로 변환


적절한 Tensor 번호로 변환을 수행해야 합니다. 이 데이터세트에는 380개의 성분이 나열되어 있으므로 `FloatTensorType`에 해당 숫자를 표기해야 합니다.


1. 380의 텐서 번호를 사용하여 변환합니다.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. onx를 만들고 **model.onnx** 파일로 저장합니다.

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

> 참고로 변환 스크립트에서 [옵션](https://onnx.ai/sklearn-onnx/parameterized.html)을 전달할 수 있습니다. 이 경우 'nocl'을 True로, 'zipmap'을 False로 전달했습니다. 이것은 분류 모델이므로 사전 목록을 생성하는 ZipMap을 제거할 수 있는 옵션이 있습니다(필수 아님). 'nocl'은 모델에 포함된 클래스 정보를 나타냅니다. 'nocl'을 'True'로 설정하여 모델의 크기를 줄입니다. 

전체 노트북을 실행하면 이제 Onnx 모델이 빌드되어 이 폴더에 저장됩니다.

## 모델 보기

Onnx 모델은 Visual Studio 코드에서 잘 보이지 않지만 많은 연구자들이 모델이 제대로 빌드되었는지 확인하기 위해 모델을 시각화하는 데 사용하는 매우 우수한 무료 소프트웨어가 있습니다. [Netron](https://github.com/lutzroeder/Netron)을 다운로드하고 model.onnx 파일을 엽니다. 380개의 입력 및 분류자가 나열된 간단한 모델을 시각화한 것을 볼 수 있습니다.

![Netron visual](images/netron.png)


Netron은 모델을 보는 데 유용한 도구입니다.

이제 웹 앱에서 이 깔끔한 모델을 사용할 준비가 되었습니다. 냉장고를 볼 때 유용할 앱을 만들고 모델에 따라 주어진 요리를 요리하는 데 사용할 수 있는 남은 재료의 조합을 알아내도록 합시다.

## 추천 웹 애플리케이션 구축

웹 앱에서 직접 모델을 사용할 수 있습니다. 이 아키텍처를 사용하면 로컬에서 실행할 수도 있고 필요한 경우 오프라인에서도 실행할 수 있습니다. `model.onnx` 파일을 저장한 동일한 폴더에 `index.html` 파일을 생성하여 시작합니다.


1. _index.html_ 파일에 다음 마크업을 추가합니다.

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```


2. 이제 `body` 태그 내에서 약간의 마크업을 추가하여 일부 구성 요소를 반영하는 체크박스 목록을 표시합니다.

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

각 확인란에는 값이 지정됩니다. 이는 데이터세트에 따라 성분이 발견된 인덱스를 반영합니다. 예를 들어, 이 알파벳 목록에서 Apple은 다섯 번째 열을 차지하므로 0부터 계산을 시작하므로 값은 '4'입니다. [성분 스프레드시트](../data/ingredient_indexes.csv)를 참조하여 다음을 찾을 수 있습니다. 주어진 성분의 색인.

  index.html 파일에서 작업을 계속하면서 마지막 닫는 `</div>` 뒤에 모델이 호출되는 스크립트 블록을 추가합니다.

1. 먼저 가져오기 [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx 런타임은 최적화 및 사용할 API를 포함하여 광범위한 하드웨어 플랫폼에서 Onnx 모델을 실행할 수 있도록 하는 데 사용됩니다.


2. 런타임이 준비되면 다음과 같이 호출할 수 있습니다.

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

이 코드에서는 몇 가지 일이 발생합니다.

1. 구성 요소 확인란이 선택되었는지 여부에 따라 설정하고 추론을 위해 모델에 보낼 380개의 가능한 값(1 또는 0)의 배열을 생성했습니다.
2. 체크박스의 배열과 애플리케이션이 시작될 때 호출되는 'init' 함수에서 체크박스가 체크되었는지 확인하는 방법을 만들었습니다. 확인란을 선택하면 '성분' 배열이 선택한 성분을 반영하도록 변경됩니다.
3. 체크박스가 선택되었는지 확인하는 'testCheckboxes' 함수를 만들었습니다.
4. 버튼을 눌렀을 때 `startInference` 기능을 사용하고, 체크박스가 체크되어 있으면 추론을 시작합니다.
5. 추론 루틴에는 다음이 포함됩니다.
   1. 모델의 비동기 로드 설정
   2. 모델에 보낼 Tensor 구조 만들기
   3. 모델을 훈련할 때 생성한 'float_input' 입력을 반영하는 '피드' 생성(Netron을 사용하여 해당 이름을 확인할 수 있음)
   4. 이 '피드'를 모델에 보내고 응답을 기다립니다.

## 애플리케이션 테스트


index.html 파일이 있는 폴더의 Visual Studio Code에서 터미널 세션을 엽니다. [http-server](https://www.npmjs.com/package/http-server)가 전역적으로 설치되어 있는지 확인하고 프롬프트에 'http-server'를 입력합니다. 로컬 호스트가 열리고 웹 앱을 볼 수 있습니다. 다양한 재료에 따라 어떤 요리가 추천되는지 확인하십시오

![ingredient web app](images/web-app.png)

축하합니다. 몇 개의 필드가 있는 '추천' 웹 앱을 만들었습니다. 시간을 내어 이 시스템을 구축하십시오!
## 🚀Challenge

웹 앱은 매우 최소화되어 있으므로 [ingredient_indexes](../data/ingredient_indexes.csv) 데이터에서 구성 요소와 해당 색인을 사용하여 계속 구축하십시오. 주어진 국가 요리를 만들기 위해 어떤 풍미 조합이 작동합니까?

## [Post-lecture quiz](https://white-water-09ec41f0f.azurestaticapps.net/quiz/26/)

## Review & Self Study

이 강의에서는 식품 재료에 대한 추천 시스템 생성의 유용성에 대해 다루었지만 ML 애플리케이션의 이 영역은 예제가 매우 풍부합니다. 이러한 시스템이 어떻게 구축되었는지 자세히 읽어보십시오.

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Assignment 

[Build a new recommender](assignment.md)
