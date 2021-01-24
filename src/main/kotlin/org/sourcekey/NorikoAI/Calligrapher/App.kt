package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.jsObject
import OpentypeJS.Font
import com.ccfraser.muirwik.components.mThemeProvider
import com.ccfraser.muirwik.components.styles.createMuiTheme
import kotlinx.browser.window
import kotlinx.css.*
import react.*
import styled.css
import styled.styledDiv


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

val wcl01Url: String = "font/HanWang/WCL-01.ttf"

val referenceFontUrl: String = "font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf"

var project: Project = Project("", referenceFontUrl)

interface AppState: RState{
    var searchUnicodeHex: String
}

class App : RComponent<RProps, AppState>() {
    override fun RBuilder.render() {
        mThemeProvider(theme = theme) {
            styledDiv {
                css {
                    position = Position.fixed
                    top = 0.px
                    left = 0.px
                    right = 0.px
                    zIndex = 1000
                }
                var timer = 0
                topBar(fun(unicodeHex){
                    window.clearTimeout(timer)
                    timer = window.setTimeout(fun(){
                        setState { searchUnicodeHex = unicodeHex }
                    }, 1000)
                })
            }
            styledDiv{ css { height = 7.vh } }
            charGrid(fun(): Font{ return project.font }, state.searchUnicodeHex)
        }
    }
}

fun RBuilder.app() = child(App::class) {}






/*
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


 */