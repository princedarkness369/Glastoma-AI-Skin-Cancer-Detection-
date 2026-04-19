from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# أسماء الفئات السبع
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
CLASS_NAMES_AR = {
    'AKIEC': 'آفة سفعية (AKIEC)',
    'BCC': 'سرطان الخلايا القاعدية (BCC)',
    'BKL': 'تقرن حميد (BKL)',
    'DF': 'ليفي جلدي (DF)',
    'MEL': 'ميلانوما - سرطان جلدي خبيث (MEL)',
    'NV': 'شامة ميلانينية (NV)',
    'VASC': 'آفة وعائية (VASC)'
}

# الألوان حسب نوع النتيجة (إيجابي/سلبي)
COLORS = {
    'malignant': '#dc2626',  # أحمر للخبيث
    'benign': '#0284c7',      # أزرق للحميد
    'warning': '#f59e0b'      # برتقالي للتحذير
}

# الإجراءات الموصى بها حسب كل نوع
RECOMMENDATIONS = {
    'AKIEC': {
        'severity': 'warning',
        'actions': [
            "🔴 استشارة طبيب جلدية فوراً - تعتبر آفة سفعية (pre-cancer)",
            "📋 إجراء خزعة لتأكيد التشخيص",
            "🧴 استخدام واقي شمس SPF 50+ يومياً",
            "❄️ يمكن علاجها بالتبريد (cryotherapy) أو الليزر",
            "📅 متابعة كل 3-6 أشهر"
        ]
    },
    'BCC': {
        'severity': 'malignant',
        'actions': [
            "🚨 حالة عاجلة - سرطان الخلايا القاعدية",
            "🩺 مراجعة طبيب جلدية خلال 48 ساعة",
            "🔪 الاستئصال الجراحي هو العلاج الأساسي",
            "💊 علاج موضعي (كريمات) في الحالات المبكرة",
            "📅 متابعة دورية كل 3 أشهر بعد العلاج"
        ]
    },
    'BKL': {
        'severity': 'benign',
        'actions': [
            "✅ حالة حميدة - تطمئن",
            "📸 متابعة دورية كل 6-12 شهراً",
            "🧴 استخدام واقي شمس SPF 30+",
            "🔍 مراقبة أي تغيرات في الحجم أو اللون",
            "❄️ يمكن إزالتها لأسباب تجميلية إذا رغبت"
        ]
    },
    'DF': {
        'severity': 'benign',
        'actions': [
            "✅ حالة حميدة (ليفي جلدي)",
            "📅 فحص دوري كل سنة",
            "👆 لا داعي للقلق - نادراً ما يتحول لخبيث",
            "🔪 يمكن إزالته جراحياً إذا سبب أعراضاً",
            "📸 توثيق بالصور للمقارنة المستقبلية"
        ]
    },
    'MEL': {
        'severity': 'malignant',
        'actions': [
            "🚨🚨 حالة خطيرة جداً - ميلانوما (سرطان جلدي خبيث)",
            "🏥 مراجعة طبيب جلدية فوراً (خلال 24 ساعة)",
            "🔪 استئصال جراحي عاجل مع هوامش أمان",
            "🧬可能需要 فحص العقد الليمفاوية",
            "📅 متابعة دقيقة كل 3 أشهر لمدة 5 سنوات",
            "🌞 حماية صارمة من الشمس مدى الحياة"
        ]
    },
    'NV': {
        'severity': 'benign',
        'actions': [
            "✅ شامة حميدة - تطمئن",
            "📸 قاعدة ABCDE لمراقبة الشامات:",
            "   • عدم تناسق (Asymmetry)",
            "   • حدود غير منتظمة (Border)",
            "   • لون غير متجانس (Color)",
            "   • قطر > 6mm (Diameter)",
            "   • تغير مع الوقت (Evolution)",
            "📅 فحص سنوي للشامات"
        ]
    },
    'VASC': {
        'severity': 'benign',
        'actions': [
            "✅ آفة وعائية حميدة (مثل الورم الوعائي)",
            "🔍 مراقبة دورية",
            "💉 يمكن علاجها بالليزر إذا رغبت",
            "📅 لا حاجة لمتابعة مكثفة",
            "🧴 حماية من الشمس للمنطقة"
        ]
    }
}

print("🔄 جاري تحميل موديل سرطان الجلد (7 أنواع)...")
try:
    model = tf.keras.models.load_model('skin_cancer_model.keras')
    print("✅ تم تحميل الموديل بنجاح!")
    print(f"📊 الفئات المدعومة: {', '.join(CLASS_NAMES)}")
except Exception as e:
    print(f"❌ خطأ في تحميل الموديل: {e}")
    model = None

def preprocess_image(image_file):
    """معالجة الصورة لتصبح جاهزة للموديل"""
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))  # تغيير حسب حجم مدخلات موديلك
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """نقطة النهاية للتنبؤ بنوع الآفة"""
    if model is None:
        return jsonify({'error': 'الموديل لم يتم تحميله بشكل صحيح'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم إرسال صورة'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'الملف فارغ'}), 400
        
        # معالجة الصورة
        processed_image = preprocess_image(file)
        
        # التنبؤ
        predictions = model.predict(processed_image)[0]
        
        # الحصول على الفئة المتوقعة والنسبة
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100
        
        # الحصول على جميع النسب لكل الفئات
        all_probabilities = {
            class_name: float(predictions[i]) * 100 
            for i, class_name in enumerate(CLASS_NAMES)
        }
        
        # تحديد شدة الحالة
        severity = RECOMMENDATIONS[predicted_class]['severity']
        
        # تحديد اللون حسب الشدة
        if severity == 'malignant':
            color = COLORS['malignant']
            result_type = 'خبيث - يستدعي التدخل الفوري'
        elif severity == 'warning':
            color = COLORS['warning']
            result_type = 'تحذير - يفضل مراجعة الطبيب'
        else:
            color = COLORS['benign']
            result_type = 'حميد - مطمئن'
        
        # تجهيز النتيجة
        result = {
            'predicted_class': predicted_class,
            'predicted_class_ar': CLASS_NAMES_AR[predicted_class],
            'confidence': round(confidence, 1),
            'severity': severity,
            'result_type': result_type,
            'color': color,
            'all_probabilities': all_probabilities,
            'message': f"🔬 النتيجة: {CLASS_NAMES_AR[predicted_class]} بنسبة {round(confidence, 1)}%",
            'recommendations': RECOMMENDATIONS[predicted_class]['actions']
        }
        
        print(f"📊 التنبؤ: {predicted_class} بنسبة {confidence:.1f}%")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """للتحقق من أن الخادم يعمل"""
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("\n🚀 تشغيل خادم كشف سرطان الجلد...")
    print(f"📍 الخادم يعمل على: http://localhost:5000")
    print(f"📋 الفئات: {', '.join(CLASS_NAMES)}")
    print("📡 انتظر حتى يتم تحميل الموديل بالكامل\n")
    app.run(host='0.0.0.0', port=5000, debug=True)