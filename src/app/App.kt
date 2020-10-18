package app

import kotlinext.js.JsObject
import kotlinext.js.Object
import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlinx.css.*
import kotlinx.html.js.onClickFunction
import org.sourcekey.NorikoAI.Calligrapher.*
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS.Glyph
import react.*
import react.dom.*
import logo.*
import org.w3c.dom.HTMLCanvasElement
import styled.css
import styled.styledDiv
import ticker.*
import kotlin.js.Json
import kotlin.math.roundToInt
import kotlin.random.Random
@JsModule("src/font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf") external val sourceHanSansTCUrl: String
@JsModule("src/font/HanWang/WCL-01.ttf") external val wcl01Url: String


var theme = null

var project: Project = Project("", wcl01Url)

class App : RComponent<RProps, RState>() {
    var timerID: Int? = null

    override fun componentDidMount() {
        timerID = window.setInterval({
            // actually, the operation is performed on a state's copy, so it stays effectively immutable
            setState{}
        }, 5000)
    }

    override fun componentWillUnmount() {
        window.clearInterval(timerID!!)
    }

    override fun RBuilder.render() {
        styledDiv {
            css {
                position = Position.fixed
                top = 0.px
                left = 0.px
                right = 0.px
                zIndex = 1000
            }
            topBar()
        }
        styledDiv{ css { height = 7.vh } }
        charGrid(project.produceFont.glyphs.glyphs.toArray())
    }
}

fun RBuilder.app() = child(App::class) {}