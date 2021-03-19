@file:JsModule("@egjs/react-infinitegrid")
@file:JsNonModule
package react.infinitegrid


import react.RClass
import react.RProps

external class OnAppendEvent{
    val currentTarget: dynamic
    val endLoading: (/*userStyle*/)->Unit
    val eventType: String
    val groupKey: Int
    val isTrusted: Boolean
    val startLoading: (/*userStyle*/)->Unit
    val stop: ()->Unit
}

external class OnPrependEvent{
    val currentTarget: dynamic
    val endLoading: (/*userStyle*/)->Unit
    val eventType: String
    val groupKey: Int
    val isTrusted: Boolean
    val startLoading: (/*userStyle*/)->Unit
    val stop: ()->Unit
}

external class OnLayoutCompleteEvent{
    val currentTarget: dynamic
    val endLoading: (/*userStyle*/)->Unit
    val eventType: String
    val fromCache: Boolean
    val isAppend: Boolean
    val isLayout: Boolean
    val isScroll: Boolean
    val isTrusted: Boolean
    val orgScrollPos: Number
    val scrollPos: Number
    val size: Number
    val stop: ()->Unit
    val target: Array<*>
}

external class OnImageErrorEvent{
    val eventType: String
}

external class OnChangeEvent{
    val eventType: String
}

external interface GridLayoutProps: RProps {
    var tag: String
    var containerTag: String
    var align: String
    var size: Int
    var margin: Int
    var isEqualSize: Boolean
    var outline: Array<*>
    var threshold: Int
    var isOverflowScroll: Boolean
    var isConstantSize: Boolean
    var useFit: Boolean
    var useRecycle: Boolean
    var useFirstRender: Boolean
    var horizontal: Boolean
    var percentage: Boolean
    var transitionDuration: Number
    var options: dynamic
    var layoutOptions: dynamic
    var layoutType: dynamic //Class
    var status: dynamic //IInfiniteGridStatus
    var loading: dynamic //React.ReactElement loading={<div>Loading...</div>}
    var onAppend: (OnAppendEvent)->Unit
    var onPrepend: (OnPrependEvent)->Unit
    var onLayoutComplete: (OnLayoutCompleteEvent)->Unit
    var onImageError: (OnImageErrorEvent)->Unit
    var onChange: (OnChangeEvent)->Unit
    var items: dynamic
    var itemBy: (dynamic)->Int
    var groupBy: (dynamic)->Int
}

@JsName("GridLayout")
external val GridLayout : RClass<GridLayoutProps>