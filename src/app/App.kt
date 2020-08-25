package app

import kotlinext.js.JsObject
import kotlinext.js.Object
import kotlinx.browser.document
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlinx.html.js.onClickFunction
import org.sourcekey.NorikoAI.Calligrapher.*
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS.Glyph
import react.*
import react.dom.*
import logo.*
import org.w3c.dom.HTMLCanvasElement
import ticker.*
import kotlin.js.Json
import kotlin.random.Random
@JsModule("src/font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf") external val sourceHanSansTCUrl: String
@JsModule("src/font/HanWang/WCL-01.ttf") external val wcl01Url: String


var project: Project = Project("", wcl01Url)

class App : RComponent<RProps, RState>() {
    override fun RBuilder.render() {
        div("App-header") {
            logo()
            h2 {
                +"Welcome to React with Kotlin"
            }
        }
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    val model = tf.sequential()
                    model.add(tf.layers.dense(jsObject{
                        inputShape = arrayOf(2, 2)
                        units = 1
                        useBias = true
                    }))
                    model.add(tf.layers.dense(jsObject{
                        units = 30
                        activation = "sigmoid"
                    }))
                    model.add(tf.layers.dense(jsObject{
                        units = 30
                        activation = "sigmoid"
                    }))
                    // Add an output layer
                    model.add(tf.layers.dense(jsObject{
                        units = 2
                        useBias = true
                    }))
                    model.compile(jsObject{
                        optimizer = tf.train.adam()
                        loss = "meanSquaredError"
                        metrics = arrayOf("mse")
                    })
                    val _model: LayersModel = model
                    val inputs = arrayOf(arrayOf(arrayOf(1, 2), arrayOf(3, 4)))
                    val inputsTensor = tf.tensor3d(inputs)!!
                    val labels = arrayOf(arrayOf(arrayOf(1, 2), arrayOf(3, 4)))
                    val labelsTensor = tf.tensor3d(labels)!!
                    _model.fit(inputsTensor, labelsTensor, jsObject {
                        this.batchSize = 32
                        this.epochs = 30
                        shuffle = true
                        callbacks = tfvis.show.fitCallbacks(
                                jsObject{ name = "Training Performance"},
                                arrayOf("loss", "mse"),
                                jsObject{
                                    height = 200
                                    callbacks = arrayOf("onEpochEnd")
                                }
                        )
                    }).await()
                    model.predict(inputsTensor)
                }
            }
            +"Test"
        }
        button {
            attrs.onClickFunction = fun(event){
                project.xx()
            }
            +"xx"
        }
        button {
            attrs.onClickFunction = fun(event){
                project.trainModel()
            }
            +"Train Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                project.saveModel()
            }
            +"Save Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                project.loadModel()
            }
            +"Load Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                println(project.produceGlyph(215, 945))
            }
            +"G"
        }
        button {
            attrs.onClickFunction = fun(event){
                console.log("Download Model")
                //model?.save("downloads://my-model")
            }
            +"Download Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                console.log("Download ModelXXXXX")
                /**/
                GlobalScope.launch {
                    println(OpentypeJS.load(sourceHanSansTCUrl).await().glyphs.glyphs.toArray()[10000].path.toPathData())
                }
                val s = "M542 721L350 721C337 756 314 802 293 839L228 818C243 789 260 752 272 721L71 721L71 653L542 653ZM418 642C405 587 377 505 355 452L194 452L247 469C239 515 216 589 190 643L128 625C153 572 174 500 180 452L44 452L44 382L270 382L270 242L66 242L66 173L270 173L270-76L343-76L343 173L545 173L545 242L343 242L343 382L567 382L567 452L424 452C446 501 470 568 490 625ZM719 11C682 11 674 20 674 70L674 831L601 831L601 72C601-30 625-58 711-58L847-58C932-58 950 0 959 164C938 169 909 183 891 197C886 49 880 11 842 11Z"
                val aGlyph = Glyph(jsObject{
                    unicode = 65
                    path = OpentypeJS.Path().setCommands(s)
                })
                /*
                val _glyphs = arrayOf(aGlyph)
                val f = OpentypeJS.Font(jsObject {
                    familyName = "null"
                    styleName = "null"
                    unitsPerEm = 1000
                    ascender = 800
                    descender = -200
                    glyphs = _glyphs
                })*/
                console.log(JSON.stringify(aGlyph))

            }
            +"Download ModelXXXX"
        }
        p("App-intro") {
            +"To get started, edit "
            code { +"app/App.kt" }
            +" and save to reload."
        }
        p("App-ticker") {
            ticker()
        }
    }
}

fun RBuilder.app() = child(App::class) {}