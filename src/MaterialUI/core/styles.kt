package MaterialUI.core

@JsModule("@material-ui/core/styles")
external object styles{
    fun fade(a: dynamic, b: Double)
    fun makeStyles(a: (theme: dynamic)->dynamic): ()->dynamic
}