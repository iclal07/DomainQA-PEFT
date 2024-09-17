# 🧠 DomainQA-PEFT: Domain-Specific Question Answering with Parameter-Efficient Fine-Tuning

**DomainQA-PEFT**, Large Language Models (LLMs) için domain-spesifik soru-cevap (QA) uygulamalarında **Parameter-Efficient Fine-Tuning (PEFT)** yöntemlerini kullanarak verimli ve hızlı bir şekilde alan adaptasyonu sağlamayı amaçlayan bir projedir.

## 📚 Proje Özeti

Bu proje, LLM'leri belirli bir alana (örneğin finans, sağlık) adapte etmek için PEFT yöntemlerinden **LoRA (Low-Rank Adaptation)** kullanarak ince ayar yapmayı hedeflemektedir. Bu sayede, daha az parametre ile etkili bir şekilde alan bilgisi edinmiş modeller oluşturabilirsiniz.

## 🚀 Başlarken

### Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız var:

```
pip install torch transformers accelerate datasets peft loralib
```

### Proje Yapısı
Proje dosya ve klasör yapısı şu şekildedir:

```
DomainQA-PEFT/
├── data/                   # 📂 Veri dosyaları ve örnek veriler
├── models/                 # 📂 Eğitilmiş modeller ve kontrol noktaları
├── src/                    # 📂 Ana kaynak kodu dosyaları
│   ├── __init__.py         # 🔧 Modül başlatma dosyası
│   ├── train.py            # 🎓 Model eğitimi ve ince ayar için ana kod
│   ├── evaluate.py         # 📊 Model değerlendirme ve test kodları
│   └── utils.py            # 🛠️ Yardımcı fonksiyonlar ve araçlar
├── scripts/                # 📂 Veri işleme ve hazırlama scriptleri
├── README.md               # 📖 Proje açıklamaları ve kullanım kılavuzu
└── requirements.txt        # 📦 Gerekli Python paketleri
```


# 📝 Kullanım Kılavuzu
## 1. 🔧 Modeli Eğitme
Modelinizi eğitmek için train.py dosyasını kullanabilirsiniz. Model eğitimi için gerekli olan kodlar ve ayarlar bu dosyada bulunmaktadır.

```
python src/train.py
```

Bu komut, varsayılan olarak Meta'nın LLaMA 2 modelini kullanarak MedQA veri kümesi üzerinde ince ayar yapar. Model eğitimi tamamlandıktan sonra models/ klasörüne kaydedilecektir.

## 2. 📊 Modeli Değerlendirme
Eğitilen modelin performansını değerlendirmek için evaluate.py dosyasını çalıştırabilirsiniz.

```
python src/evaluate.py
```

Bu komut, test veri kümesi üzerinde modeli değerlendirir ve sonuçları terminalde görüntüler.

## 3. 🛠️ Yardımcı Fonksiyonlar
utils.py dosyası, veri hazırlama, model kaydetme ve yükleme gibi sık kullanılan yardımcı işlevleri içerir. Kodunuzu daha modüler ve temiz hale getirmek için bu işlevleri kullanabilirsiniz.

📂 Dosya Açıklamaları
__init__.py: src klasörünü bir Python paketi olarak tanımlar.
train.py: PEFT kullanarak LLM'leri belirli bir alanda ince ayar yapmak için ana eğitim kodunu içerir.
evaluate.py: Eğitilen modeli değerlendirmek ve performansını ölçmek için kullanılır.
utils.py: Veri işleme, model kaydetme ve yükleme gibi yardımcı işlevleri barındırır.
🔧 Gereksinimler
Aşağıdaki dosyaları inceleyerek projenin gereksinimlerini karşılayabilirsiniz:

requirements.txt: Projede kullanılan tüm Python kütüphanelerinin listesini içerir.

```
pip install -r requirements.txt
```
### 🤝 Katkıda Bulunun
Katkıda bulunmak isterseniz, lütfen bir pull request gönderin veya bir issue açın. Her türlü katkı ve geri bildirim değerlidir!

### 📜 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasını inceleyin.

### 🙏 Teşekkürler
Bu projeye ilgi gösterdiğiniz için teşekkür ederiz! 🧠💡
