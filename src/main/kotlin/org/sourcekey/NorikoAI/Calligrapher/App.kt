package org.sourcekey.NorikoAI.Calligrapher

import kotlinx.css.*
import kotlinx.html.js.onClickFunction
import react.*
import react.dom.*
import styled.css
import styled.styledDiv
import OpentypeJS.*
import com.ccfraser.muirwik.components.mThemeProvider
import com.ccfraser.muirwik.components.styles.*
import kotlinx.browser.window

val sourceHanSansTCUrl: String = "font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf"
val wcl01Url: String = "font/HanWang/WCL-01.ttf"


var theme = createMuiTheme(jsObject {
    palettle = jsObject {
        primary = jsObject {
            light = "#757cFF"
            main = "#3f50FF"
            dark = "#0028FF"
            contrastText = "#fff"
        }
    }
}.asDynamic())

var project: Project = Project("", wcl01Url)

interface AppState: RState{
    var searchUnicodeHex: String
}

class App : RComponent<RProps, AppState>() {
    /*
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
*/
    override fun RBuilder.render() {
        console.log(theme)
        mThemeProvider(theme = theme) {
            styledDiv {
                css {
                    position = Position.fixed
                    top = 0.px
                    left = 0.px
                    right = 0.px
                    zIndex = 1000
                }
                topBar(fun(unicodeHex){
                    setState { searchUnicodeHex = unicodeHex }
                })
            }
            styledDiv{ css { height = 7.vh } }
            charGrid(fun(): OpentypeJS.Font{ return project.produceFont }, state.searchUnicodeHex)
        }
    }
}

fun RBuilder.app() = child(App::class) {}







val renderCount = functionalComponent<RProps> {
    val (count, setCount) = useState(0)
    useEffect {
        console.log("useEffect $count")
    }
    +"count $count"
    button {
        attrs {
            onClickFunction = {
                setCount(count + 1)
            }
        }
    }
}

fun RBuilder.renderCount() {
    child(renderCount)
}
